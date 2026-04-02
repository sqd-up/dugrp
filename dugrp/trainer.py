"""
DUGRP 训练器
============
实现论文 Algorithm 1（Offline Knowledge Base Construction）及三阶段训练流程。

Algorithm 1（DUGRP-Build）：
  Require: Historical data {(X_i, Y_i)}^N_i=1; radius r; temperature τ_c; queue size M; embedding dim m
  Ensure:  Knowledge base KB; HNSW index I

  1:  Init φ_θ, φ_θ̄; queue Q ← ∅
  2:  for epoch = 1 to E do
  3:    for each anchor X_i do
  4:      Sample X_i^+ ~ Uniform(N_r(i))         ▷ r-邻域正样本
  5:      q ← φ_θ(X_i); k^+ ← φ_θ̄(X_i^+)
  6:      Compute L_NCE with q, k^+, Q            ▷ Eq.3
  7:      Update θ ← θ - η∇L_NCE
  8:      EMA: θ̄ ← m·θ̄ + (1-m)·θ
  9:      Enqueue k^+; dequeue oldest if |Q| > M
  10:   end for
  11: end for
  12: z_i ← φ_θ(X_i) for all i ∈ [N]
  13: KB ← {(z_i, X_i, Y_i)}^N_i=1              ▷ O(N log N)
  14: I ← BuildHNSW({z_i})
  15: return KB, I

训练总损失（Section 4.3）：
  L_total = L_pred + λ_contrastive · L_NCE
"""

from __future__ import annotations

import logging
import os
import random
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from .config import DUGRPConfig
from .encoder import ContrastiveEncoder
from .predictor import DUGRPPredictor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 对比学习数据集（Algorithm 1 line 3-4）
# ---------------------------------------------------------------------------
class ContrastiveDataset(Dataset):
    """
    为 Algorithm 1 提供 (anchor, positive) 对。

    正样本采样策略（Algorithm 1 line 4）：
      X_i^+ ~ Uniform(N_r(i))
      N_r(i) = {j : ||z_i - z_j||² ≤ r}（r-邻域）

    实现简化：
      - 若 use_temporal_proximity=True：从时间邻近窗口内随机采样（近似 r-邻域）
      - 若 use_temporal_proximity=False：对同一序列加入高斯噪声作为正样本（自增广）
    """

    def __init__(
        self,
        sequences: np.ndarray,          # (N, T, D)
        context_length: int,
        prediction_length: int,
        neighbor_window: int = 30,      # 时间邻域窗口半径（增大以提高召回）
        augment_noise_std: float = 0.01,
        use_temporal_proximity: bool = True,
        delay_aug_steps: int = 0,       # >0时启用延迟增广：以delay_aug_steps时间步偏移作正样本
                                        # 模拟"delayed context ≈ clean context"，解决测试分布偏移
    ):
        super().__init__()
        self.sequences = sequences.astype(np.float32)
        self.L = context_length
        self.H = prediction_length
        self.neighbor_window = neighbor_window
        self.noise_std = augment_noise_std
        self.use_temporal_proximity = use_temporal_proximity
        self.delay_aug_steps = delay_aug_steps  # 延迟增广步数

        # 有效锚点索引：需要足够长的序列
        min_len = context_length + prediction_length
        self.valid_indices = [
            (i, t)
            for i, seq in enumerate(sequences)
            for t in range(context_length, len(seq) - prediction_length + 1)
        ]

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回：(X_anchor, X_positive, Y_anchor, Y_positive)
          X_anchor:   (L, D)
          X_positive: (L, D)
          Y_anchor:   (H, D)
          Y_positive: (H, D)
        """
        seq_idx, t = self.valid_indices[idx]
        seq = self.sequences[seq_idx]  # (T_total, D)
        T_total = len(seq)

        # 锚点上下文和未来
        X_anchor = seq[t - self.L: t]               # (L, D)
        Y_anchor = seq[t: t + self.H]               # (H, D)

        if self.delay_aug_steps > 0:
            # 延迟增广正样本：将锚点向过去平移 delay_aug_steps 步
            # 含义："延迟d步后观测到的上下文" ≈ "当前真实上下文"
            # 这使编码器学会：delayed_context 和 clean_context 在嵌入空间中相近
            # 解决测试时查询(delayed)与KB(clean)的分布不匹配问题
            d_steps = random.randint(1, self.delay_aug_steps)
            t_anc_delayed = max(self.L, t - d_steps)
            X_positive = seq[t_anc_delayed - self.L: t_anc_delayed]   # delayed version
            Y_positive = seq[t_anc_delayed: t_anc_delayed + self.H]
        elif self.use_temporal_proximity:
            # 从时间邻域 [t-W, t+W] 内采样正样本起点（Algorithm 1 line 4）
            low  = max(self.L, t - self.neighbor_window)
            high = min(T_total - self.H, t + self.neighbor_window)
            if low >= high:
                t_pos = t
            else:
                t_pos = random.randint(low, high - 1)
            X_positive = seq[t_pos - self.L: t_pos]  # (L, D)
            Y_positive = seq[t_pos: t_pos + self.H]  # (H, D)
        else:
            # 加噪声自增广
            noise = np.random.randn(*X_anchor.shape).astype(np.float32) * self.noise_std
            X_positive = X_anchor + noise
            Y_positive = Y_anchor.copy()

        return (
            torch.from_numpy(X_anchor),
            torch.from_numpy(X_positive),
            torch.from_numpy(Y_anchor),
            torch.from_numpy(Y_positive),
        )


# ---------------------------------------------------------------------------
# DUGRP 训练器主类
# ---------------------------------------------------------------------------
class DUGRPTrainer:
    """
    三阶段训练流程：

    Phase 1：对比预训练（Algorithm 1，离线）
      - 目标：训练 φ_θ 使拓扑分离边界 Δ > 0（Theorem 1）
      - 损失：L_NCE（Eq.3）

    Phase 2：知识库构建（Algorithm 1 line 12-15）
      - 用训练集全量数据计算嵌入并填充知识库
      - 构建 HNSW 索引（O(N log N)）

    Phase 3：端到端微调（可选，如需适配特定场景）
      - 在验证集上进行少量在线更新
    """

    def __init__(self, config: DUGRPConfig, predictor: DUGRPPredictor):
        self.cfg = config
        self.predictor = predictor
        self.device = torch.device(config.device)

    # -----------------------------------------------------------------------
    # Phase 1：对比预训练（Algorithm 1 lines 1-11）
    # -----------------------------------------------------------------------
    def train_contrastive(
        self,
        train_sequences: np.ndarray,   # (N, T, D)
        val_sequences: Optional[np.ndarray] = None,
        checkpoint_dir: str = "checkpoints/dugrp",
        log_interval: int = 50,
    ) -> List[float]:
        """
        执行 Algorithm 1 的对比预训练阶段（lines 1-11）。

        返回：每个 epoch 的训练损失列表。
        """
        cfg = self.cfg
        encoder = self.predictor.encoder
        encoder.train()

        # 数据集与 DataLoader
        # delay_aug_steps=5：模拟最多5步（500ms@10Hz）的延迟正样本对，覆盖D3最大延迟(340ms≈4步)
        # 使编码器学会：delayed context ≈ clean context in embedding space
        dataset = ContrastiveDataset(
            sequences=train_sequences,
            context_length=cfg.context_length,
            prediction_length=cfg.prediction_length,
            neighbor_window=cfg.neighbor_window,
            use_temporal_proximity=True,
            delay_aug_steps=5,
        )
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=(cfg.device == "cuda"),
            drop_last=True,
        )

        # 优化器（仅优化在线编码器，动量编码器由 EMA 更新）
        optimizer = optim.AdamW(
            list(encoder.encoder.parameters()) + list(encoder.projection.parameters()),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs_contrastive
        )

        epoch_losses: List[float] = []

        for epoch in range(1, cfg.epochs_contrastive + 1):
            epoch_loss = 0.0
            n_batches = 0

            for batch_idx, (X_anc, X_pos, _, _) in enumerate(loader):
                X_anc = X_anc.to(self.device)   # (B, L, D)
                X_pos = X_pos.to(self.device)   # (B, L, D)

                optimizer.zero_grad()
                # Algorithm 1 lines 5-9（Eq.3 InfoNCE + EMA + 队列更新）
                loss = encoder.contrastive_loss(X_anc, X_pos)
                loss.backward()
                # 梯度裁剪（防止 MoCo 早期训练不稳定）
                nn.utils.clip_grad_norm_(encoder.encoder.parameters(), max_norm=1.0)
                nn.utils.clip_grad_norm_(encoder.projection.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches  += 1

                if (batch_idx + 1) % log_interval == 0:
                    logger.info(
                        f"[Phase1] Epoch {epoch}/{cfg.epochs_contrastive} | "
                        f"Batch {batch_idx+1}/{len(loader)} | "
                        f"L_NCE={loss.item():.4f}"
                    )

            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)
            epoch_losses.append(avg_loss)
            logger.info(
                f"[Phase1] Epoch {epoch}/{cfg.epochs_contrastive} | "
                f"avg L_NCE={avg_loss:.4f} | "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

            # 保存最佳 checkpoint
            if epoch == 1 or avg_loss <= min(epoch_losses):
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(
                    encoder.state_dict(),
                    os.path.join(checkpoint_dir, "encoder_best.pt"),
                )

        encoder.eval()
        logger.info("[Phase1] 对比预训练完成。")
        return epoch_losses

    # -----------------------------------------------------------------------
    # Phase 2：知识库构建（Algorithm 1 lines 12-15）
    # -----------------------------------------------------------------------
    def build_knowledge_base(
        self,
        train_sequences: np.ndarray,   # (N, T, D)
        prediction_length: Optional[int] = None,
        batch_size: int = 256,
    ):
        """
        用训练集全量数据构建知识库（Algorithm 1 lines 12-15）。

        z_i ← φ_θ(X_i)
        KB  ← {(z_i, X_i, Y_i)}^N_i=1
        I   ← BuildHNSW({z_i})   ▷ O(N log N)

        参数
        ----
        train_sequences : (N, T_total, D) — 训练集完整序列
        prediction_length : H（默认使用 config 值）
        """
        cfg = self.cfg
        H = prediction_length or cfg.prediction_length
        L = cfg.context_length

        # 提取所有 (context, future) 对
        all_contexts: List[np.ndarray] = []
        all_futures:  List[np.ndarray] = []

        for seq in train_sequences:
            T_total = len(seq)
            # 滑动窗口，步长=1
            for t in range(L, T_total - H + 1):
                all_contexts.append(seq[t - L: t])     # (L, D)
                all_futures.append(seq[t: t + H])       # (H, D)

        contexts_arr = np.stack(all_contexts, axis=0)   # (M, L, D)
        futures_arr  = np.stack(all_futures,  axis=0)   # (M, H, D)

        logger.info(
            f"[Phase2] 提取 {len(contexts_arr)} 个 (context, future) 对，"
            f"开始计算嵌入..."
        )
        self.predictor.add_batch_to_kb(contexts_arr, futures_arr, batch_size=batch_size)
        logger.info(
            f"[Phase2] 知识库构建完成，大小：{self.predictor.kb.size}。"
            f"HNSW 索引将在首次检索时自动构建。"
        )

    # -----------------------------------------------------------------------
    # 完整训练入口
    # -----------------------------------------------------------------------
    def fit(
        self,
        train_sequences: np.ndarray,   # (N, T, D)
        val_sequences: Optional[np.ndarray] = None,
        checkpoint_dir: str = "checkpoints/dugrp",
    ) -> dict:
        """
        执行完整的 Algorithm 1：
          Phase 1 → 对比预训练
          Phase 2 → 知识库构建

        返回：训练统计信息。
        """
        logger.info("=" * 60)
        logger.info("[DUGRP-Build] 开始 Phase 1：对比预训练...")
        logger.info("=" * 60)
        losses = self.train_contrastive(
            train_sequences=train_sequences,
            val_sequences=val_sequences,
            checkpoint_dir=checkpoint_dir,
        )

        logger.info("=" * 60)
        logger.info("[DUGRP-Build] 开始 Phase 2：知识库构建...")
        logger.info("=" * 60)
        self.build_knowledge_base(train_sequences=train_sequences)

        # 保存完整 checkpoint
        self.predictor.save(checkpoint_dir)

        return {
            "contrastive_losses": losses,
            "final_loss": losses[-1] if losses else None,
            "kb_size": self.predictor.kb.size,
        }
