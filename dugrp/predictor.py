"""
DUGRP 在线流式预测器
====================
实现论文 Algorithm 2（Online DUGRP Streaming Prediction）。

Algorithm 2：
  Require: Delayed context X_t^τ; delay τ(t); prior uncertainty U_{t-1}; Chronos-2 F_C
  Ensure:  Prediction Ŷ_{t+1:t+H}; updated U_t

  1: z_t ← φ_θ(X_t^τ)                    ▷ Theorem 1（对比编码）
  2: K  ← K*(τ(t)) via Eq.11             ▷ Theorem 2（动态检索量）
  3: KB_K ← I.SEARCH(z_t, K)             ▷ O(log N) HNSW 检索
  4: α  ← α(U_{t-1}, τ(t)) via Eq.12    ▷ 不确定性融合权重
  5: Compute {w_k} via Eq.13             ▷ Softmax 检索权重
  6: X_ret ← Σ w_k X_k^ret              ▷ 加权检索均值
     X_fused ← (1-α)X_t^τ + α X_ret     ▷ Eq.14 自适应融合
  7: Ŷ_{t+1:t+H}, {Q_q} ← F_C(X_fused) ▷ Chronos-2 推理
  8: U_t ← (Q_{0.9} - Q_{0.1})/|Ŷ|     ▷ 闭环不确定性更新
  9: return Ŷ, U_t

总端到端延迟：编码0.4ms + 检索2.1ms + 融合0.2ms + Chronos-2 7.8ms ≈ 11.5ms < Δt_max=12ms
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .config import DUGRPConfig
from .encoder import ContrastiveEncoder
from .knowledge_base import KnowledgeBase, RetrievalResult
from .dynamic_retrieval import (
    compute_K_star,
    compute_alpha,
    compute_retrieval_weights,
    fuse_context,
    update_uncertainty,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 预测结果数据结构
# ---------------------------------------------------------------------------
@dataclass
class PredictionResult:
    """DUGRP 完整预测结果（对应 Algorithm 2 输出）。"""

    # --- Algorithm 2 主要输出 ---
    y_pred:     np.ndarray              # (H, D)  — 点预测（Q_{0.5}）
    q_low:      np.ndarray              # (H, D)  — Q_{0.1} 分位数
    q_high:     np.ndarray              # (H, D)  — Q_{0.9} 分位数
    uncertainty: float                  # U_t（Algorithm 2 line 8）

    # --- 中间状态（可用于调试/消融） ---
    K:          int                     # 本步检索量 K*(τ)
    alpha:      float                   # 本步融合权重 α
    tau:        float                   # 当前时延 (ms)
    x_fused:    Optional[np.ndarray] = None  # (T, D) — 融合后上下文
    retrieval:  Optional[RetrievalResult] = None  # 检索结果（含 scores）


# ---------------------------------------------------------------------------
# DUGRP 预测器主类
# ---------------------------------------------------------------------------
class DUGRPPredictor:
    """
    DUGRP 在线流式预测器。

    生命周期：
      1. __init__   — 初始化编码器、知识库、Chronos-2
      2. add_to_kb  — 离线阶段：向知识库添加历史数据（Algorithm 1 产出）
      3. predict    — 在线阶段：Algorithm 2 逐步调用

    参数
    ----
    config : DUGRPConfig — 所有超参数（严格对应论文）
    """

    def __init__(self, config: DUGRPConfig):
        self.cfg = config
        self.device = torch.device(config.device)

        # ----------------------------------------------------------------
        # 1. 对比学习编码器（Innovation I，Section 4.1）
        # ----------------------------------------------------------------
        self.encoder = ContrastiveEncoder(
            input_dim=config.input_dim,
            embed_dim=config.embed_dim,
            temperature=config.temperature,
            queue_size=config.queue_size,
            momentum=config.momentum,
        ).to(self.device)
        self.encoder.eval()

        # ----------------------------------------------------------------
        # 2. HNSW 知识库（Section 4.1.3）
        # ----------------------------------------------------------------
        self.kb = KnowledgeBase(
            embed_dim=config.embed_dim,
            hnsw_M=config.hnsw_M,
            hnsw_ef_construction=config.hnsw_ef_construction,
            hnsw_ef_search=config.hnsw_ef_search,
        )

        # ----------------------------------------------------------------
        # 3. Chronos-2（Section 4.4）
        # 懒加载：仅在首次 predict() 时加载，避免启动时占用显存
        # ----------------------------------------------------------------
        self._chronos = None
        self._chronos_model_id = config.chronos_model_id

        # ----------------------------------------------------------------
        # 4. 闭环不确定性状态（Algorithm 2 line 8）
        # ----------------------------------------------------------------
        self._U_prev: float = 0.5   # 初始不确定性（中性值）

        logger.info(
            f"[DUGRPPredictor] 初始化完成 | device={config.device} | "
            f"embed_dim={config.embed_dim} | K_min={config.K_min} | K_max={config.K_max}"
        )

    # -----------------------------------------------------------------------
    # Chronos-2 懒加载
    # -----------------------------------------------------------------------
    def _get_chronos(self):
        """懒加载 Chronos-2 pipeline，并尝试 torch.compile 加速。"""
        if self._chronos is None:
            logger.info(f"[DUGRPPredictor] 加载 Chronos-2：{self._chronos_model_id}")
            try:
                from chronos.chronos2 import Chronos2Pipeline
                self._chronos = Chronos2Pipeline.from_pretrained(
                    self._chronos_model_id,
                    device_map=str(self.device),
                )
            except Exception as e:
                raise RuntimeError(
                    f"无法加载 Chronos-2 模型 '{self._chronos_model_id}'。\n"
                    f"请确认：pip install 'chronos-forecasting @ chronos-forecasting-main'\n"
                    f"原始错误：{e}"
                )

            # torch.compile：PyTorch 2.0+ 在 CUDA 上可提供 2-3x 推理加速
            # reduce-overhead 模式启用 CUDA graphs，消除 Python 调度开销
            if self.device.type == "cuda":
                try:
                    import torch
                    if hasattr(torch, "compile"):
                        self._chronos.model = torch.compile(
                            self._chronos.model,
                            mode="reduce-overhead",
                            fullgraph=False,  # 允许局部不支持的op回退
                        )
                        logger.info("[DUGRPPredictor] Chronos-2 已通过 torch.compile 优化")
                except Exception as e:
                    logger.warning(f"[DUGRPPredictor] torch.compile 跳过（{e}）")
        return self._chronos

    # -----------------------------------------------------------------------
    # 知识库填充（离线阶段，Algorithm 1 产出）
    # -----------------------------------------------------------------------
    def add_to_kb(
        self,
        context: np.ndarray,   # (T_context, D)
        future: np.ndarray,    # (T_future, D)
    ):
        """
        计算嵌入并加入知识库（Algorithm 1 line 12-13）。
        z_i ← φ_θ(X_i)
        KB ← {(z_i, X_i, Y_i)}
        """
        emb = self._embed(context)
        self.kb.add(context, future, emb)

    def add_batch_to_kb(
        self,
        contexts: np.ndarray,   # (N, T_context, D)
        futures:  np.ndarray,   # (N, T_future, D)
        batch_size: int = 256,
    ):
        """批量填充知识库（内存友好）。"""
        N = len(contexts)
        embeddings = np.zeros((N, self.cfg.embed_dim), dtype=np.float32)

        self.encoder.eval()
        with torch.no_grad():
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                batch = torch.from_numpy(
                    contexts[start:end].astype(np.float32)
                ).to(self.device)
                embs = self.encoder.encode(batch)  # (B, m)
                embeddings[start:end] = embs.cpu().numpy()

        self.kb.add_batch(contexts, futures, embeddings)
        logger.info(f"[DUGRPPredictor] 知识库批量加入 {N} 条，当前大小：{self.kb.size}")

    # -----------------------------------------------------------------------
    # 单条嵌入（推理用）
    # -----------------------------------------------------------------------
    def _embed(self, context: np.ndarray) -> np.ndarray:
        """
        x: (T, D) → z: (m,)，L2 归一化。
        对应 Algorithm 2 line 1：z_t ← φ_θ(X_t^τ)
        """
        self.encoder.eval()
        with torch.no_grad():
            x = torch.from_numpy(
                context.astype(np.float32)
            ).unsqueeze(0).to(self.device)      # (1, T, D)
            z = self.encoder.encode(x)          # (1, m)
            return z.squeeze(0).cpu().numpy()   # (m,)

    # -----------------------------------------------------------------------
    # 核心预测（Algorithm 2）
    # -----------------------------------------------------------------------
    def predict(
        self,
        context: np.ndarray,      # (T_context, D)：当前时延上下文 X_t^τ
        tau: float,               # 当前时延 τ(t)，单位 ms
        use_retrieval: bool = True,
    ) -> PredictionResult:
        """
        执行 Algorithm 2 的完整流程。

        参数
        ----
        context       : (T, D) — 当前可观测的时延上下文
        tau           : 当前时延 (ms)
        use_retrieval : 是否使用知识库检索（消融实验用）

        返回
        ----
        PredictionResult（含点预测、分位数、不确定性及中间状态）
        """
        cfg = self.cfg

        # --- Algorithm 2 line 1：编码当前上下文 ---
        z_t = self._embed(context)    # (m,)

        # --- Algorithm 2 line 2：K*(τ)（Eq.11） ---
        K = compute_K_star(
            tau=tau,
            K_0=cfg.K_0,
            beta_tau=cfg.beta_tau,
            K_min=cfg.K_min,
            K_max=cfg.K_max,
        )

        # --- Algorithm 2 line 4：α(U_{t-1}, τ)（Eq.12） ---
        alpha = compute_alpha(
            U_t=self._U_prev,
            tau=tau,
            gamma1=cfg.gamma1,
            gamma2=cfg.gamma2,
            tau_min=cfg.tau_min,
            tau_max=cfg.tau_max,
        )

        # --- Algorithm 2 line 3：HNSW 检索 ---
        retrieval: Optional[RetrievalResult] = None
        if use_retrieval and self.kb.size > 0:
            retrieval = self.kb.search(z_t, K)

        # --- Algorithm 2 lines 5-6：融合上下文（Eq.13 + Eq.14） ---
        if retrieval is not None and len(retrieval.sequences) > 0:
            w = compute_retrieval_weights(retrieval.scores, cfg.temperature)
            X_fused = fuse_context(context, retrieval.sequences, w, alpha)
        else:
            # 无检索时退化为纯时延上下文（等价于 DUGRP-noKB 消融）
            X_fused = context.copy()

        # --- Algorithm 2 line 7：Chronos-2 推理 ---
        q_low, q_mid, q_high = self._chronos_predict(X_fused)

        # ---------------------------------------------------------------
        # AR-Chronos Horizon-Adaptive 融合
        # 核心洞察：Chronos-2 是通用大模型，对短期线性趋势反应迟钝；
        #   线性 AR 在短 horizon（1-5步）天然精准（速度近似匀速）。
        #   指数衰减权重：AR 主导近期，Chronos-2 主导远期。
        #
        # ar_weight(h) = exp(-h / λ)
        #   h=1: ~0.85  → 85% AR（近未来）
        #   h=6: ~0.37  → 37% AR
        #   h=12: ~0.14 → 14% AR
        #   h=24: ~0.02 → 2%  AR（远未来，Chronos-2 主导）
        # ---------------------------------------------------------------
        if cfg.use_ar_blend:
            q_low_ar, q_mid_ar, q_high_ar = self._ar_predict(context)   # 使用原始上下文

            H_pred = q_mid.shape[0]
            h = np.arange(1, H_pred + 1, dtype=np.float32).reshape(-1, 1)   # (H, 1)
            ar_w = np.exp(-h / cfg.ar_blend_horizon)                          # (H, 1)

            q_mid  = ar_w * q_mid_ar  + (1.0 - ar_w) * q_mid
            q_low  = ar_w * q_low_ar  + (1.0 - ar_w) * q_low
            q_high = ar_w * q_high_ar + (1.0 - ar_w) * q_high

        # ---------------------------------------------------------------
        # KB 未来融合（Innovation IV：RAG-for-TimeSeries）
        # 核心思想：若检索到的历史时刻在运动模式上与当前高度相似，
        #   则该时刻的"接下来发生了什么(Y_k^future)"是当前预测的最强参考。
        #   这是纯 Chronos-2 无法提供的领域特异性先验。
        #
        # 只移动预测区间中心，保持区间宽度不变，从而：
        #   ① 降低 MAE：q_mid 向真实未来靠近
        #   ② 改善 CRPS：区间随中心平移，覆盖率维持
        # ---------------------------------------------------------------
        if (use_retrieval
                and retrieval is not None
                and len(retrieval.futures) > 0
                and cfg.blend_kb_futures):
            H_pred, D_pred = q_mid.shape
            w_fut = compute_retrieval_weights(retrieval.scores, cfg.temperature)

            # 对齐 K 条检索未来至预测步长 H
            kb_futures = np.zeros(
                (len(retrieval.futures), H_pred, D_pred), dtype=np.float32
            )
            for k, fut in enumerate(retrieval.futures):
                H_k = min(fut.shape[0], H_pred)
                kb_futures[k, :H_k, :] = fut[:H_k, :]

            Y_kb = np.einsum('k,khd->hd', w_fut, kb_futures)  # (H, D)

            # 混合系数：与相似度成正比，无硬阈值
            # 原版用 threshold=0.3：当编码器未充分训练时分数普遍<0.3，导致blend_coef=0完全失效。
            # 新版：blend_coef = clip(score, 0, 1) × kb_blend_max
            # 分数0.1→blend 6%，分数0.5→blend 30%，分数0.9→blend 54%（温和且始终激活）
            mean_score = float(np.mean(np.clip(retrieval.scores, 0.0, 1.0)))
            blend_coef = float(mean_score * cfg.kb_blend_max)

            if blend_coef > 0.0:
                old_q_mid = q_mid.copy()
                q_mid  = (1.0 - blend_coef) * old_q_mid + blend_coef * Y_kb
                delta  = q_mid - old_q_mid  # 区间整体平移量
                q_low  = q_low  + delta     # 保持区间宽度，只移动位置
                q_high = q_high + delta

        # --- Algorithm 2 line 8：闭环不确定性更新 ---
        U_t = update_uncertainty(q_low, q_high, q_mid)
        self._U_prev = U_t   # 存储供下一步使用

        return PredictionResult(
            y_pred=q_mid,
            q_low=q_low,
            q_high=q_high,
            uncertainty=U_t,
            K=K,
            alpha=alpha,
            tau=tau,
            x_fused=X_fused,
            retrieval=retrieval,
        )

    # -----------------------------------------------------------------------
    # 线性自回归预测（AR baseline，用于 horizon-adaptive 融合）
    # -----------------------------------------------------------------------
    def _ar_predict(
        self,
        X_context: np.ndarray,   # (T, D) — 观测上下文（已归一化速度信号）
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        基于最近 n_fit 步的线性趋势外推，生成 H 步预测。

        物理动机：
          ROV 速度信号在短时间内（<1s）近似匀速运动。
          对于前3-8步（0.3-0.8s），线性外推精度优于 Chronos-2（Chronos-2
          依赖大量上下文，对短期趋势反应迟钝）。
          horizon-adaptive blend 将在短步距使用 AR，长步距回归 Chronos-2。

        不确定性估计：
          σ_h = σ_resid × √h（随机游走假设，不确定性随预测步长增大）
          对应 80% 覆盖区间：[q_mid - 1.28σ_h, q_mid + 1.28σ_h]

        返回：(q_low, q_mid, q_high)，每个形状 (H, D)
        """
        H = self.cfg.prediction_length
        T, D = X_context.shape

        # 拟合最近 n_fit 步（太短噪声大，太长包含旧趋势）
        n_fit = min(8, T)
        X_recent = X_context[-n_fit:]                      # (n_fit, D)
        t_fit = np.arange(n_fit, dtype=np.float64)

        slopes = np.zeros(D, dtype=np.float64)
        intercepts = np.zeros(D, dtype=np.float64)
        resid_stds = np.zeros(D, dtype=np.float64)

        for d in range(D):
            p = np.polyfit(t_fit, X_recent[:, d].astype(np.float64), 1)
            slopes[d] = p[0]
            intercepts[d] = p[1]
            fitted = np.polyval(p, t_fit)
            resid_stds[d] = np.std(X_recent[:, d] - fitted) + 1e-6

        # 外推到未来 H 步（相对于最后一个拟合点的步长编号）
        t_future = np.arange(n_fit, n_fit + H, dtype=np.float64)   # (H,)
        q_mid = (slopes[None, :] * t_future[:, None]
                 + intercepts[None, :]).astype(np.float32)           # (H, D)

        # 随预测步长增大的不确定性
        h_idx = np.arange(1, H + 1, dtype=np.float32).reshape(-1, 1)  # (H, 1)
        sigma = np.sqrt(h_idx) * resid_stds[None, :].astype(np.float32)  # (H, D)
        q_low  = q_mid - 1.28 * sigma
        q_high = q_mid + 1.28 * sigma

        return q_low, q_mid, q_high

    # -----------------------------------------------------------------------
    # Chronos-2 推理封装（Algorithm 2 line 7）
    # -----------------------------------------------------------------------
    def _chronos_predict(
        self,
        X_fused: np.ndarray,   # (T, D)
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        调用 Chronos-2 predict_quantiles，返回三个分位数预测。

        输入格式（Chronos-2 多变量 dict-based）：
          {"target": (D, T)}  — 将 (T, D) 转置

        输出：
          q_low  : (H, D) — Q_{0.1}
          q_mid  : (H, D) — Q_{0.5}（点预测）
          q_high : (H, D) — Q_{0.9}

        论文：Chronos-2 输出 inter-quantile range 作为不确定性代理（Section 4.3.1）
        """
        chronos = self._get_chronos()

        T, D = X_fused.shape
        target = X_fused.T.astype(np.float32)    # (D, T)

        chronos_input = {"target": target}

        quantile_levels = self.cfg.quantile_levels  # [0.1, 0.5, 0.9]

        # fp16 autocast：在支持的GPU上将推理时间减半（A100/V100/RTX等均支持）
        use_amp = (self.device.type == "cuda")
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=use_amp):
                quantiles_list, mean_list = chronos.predict_quantiles(
                    [chronos_input],
                    prediction_length=self.cfg.prediction_length,
                    quantile_levels=quantile_levels,
                )

        # quantiles_list[0]: (D, H, n_q)
        # mean_list[0]:      (D, H)
        q_tensor = quantiles_list[0].cpu().numpy()   # (D, H, 3)
        # mean 实际是 Q_{0.5}（median），与 q_tensor[...,1] 相同

        # 映射分位数索引
        q_idx = {q: i for i, q in enumerate(quantile_levels)}

        q_low  = q_tensor[:, :, q_idx[0.1]].T    # (H, D)
        q_mid  = q_tensor[:, :, q_idx[0.5]].T    # (H, D)
        q_high = q_tensor[:, :, q_idx[0.9]].T    # (H, D)

        return q_low, q_mid, q_high

    # -----------------------------------------------------------------------
    # 预热（评估前调用，消除Chronos-2首次加载和GPU JIT编译带来的延迟尖峰）
    # -----------------------------------------------------------------------
    def warmup(self, n_calls: int = 10):
        """
        运行 n_calls 次伪预测以预热 Chronos-2 和 GPU。
        在正式评估计时前调用，消除冷启动延迟（可达100ms+）。
        torch.compile 需要多次调用才能完成 CUDA graph 的 trace 和编译，
        建议 n_calls=10，确保编译完成后延迟稳定在目标范围内。
        """
        logger.info(f"[DUGRPPredictor] 预热 Chronos-2（{n_calls}次）...")
        dummy = np.zeros((self.cfg.context_length, self.cfg.input_dim), dtype=np.float32)
        dummy[:, 0] = np.sin(np.linspace(0, 4 * np.pi, self.cfg.context_length))
        for _ in range(n_calls):
            self._chronos_predict(dummy)
        self.reset_uncertainty()
        logger.info("[DUGRPPredictor] 预热完成。")

    # -----------------------------------------------------------------------
    # 状态重置（评估不同场景时使用）
    # -----------------------------------------------------------------------
    def reset_uncertainty(self, U_init: float = 0.5):
        """重置闭环不确定性状态（每次评估场景切换时调用）。"""
        self._U_prev = U_init

    # -----------------------------------------------------------------------
    # 持久化
    # -----------------------------------------------------------------------
    def save(self, checkpoint_dir: str):
        """保存编码器权重和知识库。"""
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(
            self.encoder.state_dict(),
            os.path.join(checkpoint_dir, "encoder.pt"),
        )
        self.kb.save(os.path.join(checkpoint_dir, "knowledge_base"))
        logger.info(f"[DUGRPPredictor] 已保存至 {checkpoint_dir}")

    def load(self, checkpoint_dir: str):
        """加载编码器权重和知识库。"""
        import os
        encoder_path = os.path.join(checkpoint_dir, "encoder.pt")
        self.encoder.load_state_dict(
            torch.load(encoder_path, map_location=self.device)
        )
        self.encoder.eval()
        kb_path = os.path.join(checkpoint_dir, "knowledge_base")
        self.kb = KnowledgeBase.load(kb_path)
        logger.info(
            f"[DUGRPPredictor] 已加载 {checkpoint_dir}，KB 大小：{self.kb.size}"
        )
