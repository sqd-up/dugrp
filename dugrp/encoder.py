"""
DUGRP Innovation I：拓扑感知对比知识库编码器
=============================================
论文 Section 4.1 / Theorem 1

核心组件：
  - ContrastiveEncoder：4层时序 CNN，输出 S^{m-1} 上的 L2 归一化嵌入
  - MoCo 动量编码器：负样本队列维护（M=65536，EMA 动量=0.99）
  - InfoNCE 损失（Eq.3）

Theorem 1 保证：
  对齐误差 E[||φ(x)-φ(x⁺)||²] ≤ 2·L²·C_τ·r/τ_c
  分离边界 E[⟨φ(x),φ(x̃)⟩] ≤ 1 - Δ，Δ > 0
  → 可用 O(log N) HNSW 实现精确检索
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 内部：4层时序 CNN 主干
# ---------------------------------------------------------------------------
def _build_temporal_cnn(input_dim: int, embed_dim: int) -> nn.Sequential:
    """
    4层时序 CNN（论文 Section 5.1.4）。
    输入：(B, D, T)  → 输出：(B, embed_dim)

    层设计：
      Conv1(D   → 64,  k=7)  → BN → ReLU → MaxPool(2)
      Conv2(64  → 128, k=5)  → BN → ReLU → MaxPool(2)
      Conv3(128 → 256, k=3)  → BN → ReLU → MaxPool(2)
      Conv4(256 → 512, k=3)  → BN → ReLU → AdaptiveAvgPool → Flatten
    共4层卷积，与论文"4-layer temporal CNN"一致。
    """
    return nn.Sequential(
        # Layer 1
        nn.Conv1d(input_dim, 64, kernel_size=7, padding=3, bias=False),
        nn.BatchNorm1d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(kernel_size=2),
        # Layer 2
        nn.Conv1d(64, 128, kernel_size=5, padding=2, bias=False),
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(kernel_size=2),
        # Layer 3
        nn.Conv1d(128, 256, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(kernel_size=2),
        # Layer 4
        nn.Conv1d(256, 512, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),  # (B, 512)
    )


def _build_projection_head(embed_dim: int) -> nn.Sequential:
    """
    投影头：512 → m（2层 MLP，与 MoCo v2 设计一致）。
    """
    return nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(inplace=True),
        nn.Linear(512, embed_dim),
    )


# ---------------------------------------------------------------------------
# 主类：ContrastiveEncoder
# ---------------------------------------------------------------------------
class ContrastiveEncoder(nn.Module):
    """
    MoCo 风格对比编码器（论文 Section 4.1）。

    参数
    ----
    input_dim  : D，输入信号维度（6-DOF）
    embed_dim  : m，球面嵌入维度（论文 m=128）
    temperature: τ_c，InfoNCE 温度（论文 τ_c=0.07）
    queue_size : M，负样本队列大小（论文 M=65536）
    momentum   : EMA 系数（论文 Algorithm 1 line 8，值=0.99）
    """

    def __init__(
        self,
        input_dim: int = 6,
        embed_dim: int = 128,
        temperature: float = 0.07,
        queue_size: int = 65536,
        momentum: float = 0.99,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.temperature = temperature
        self.queue_size = queue_size
        self.momentum = momentum

        # --- 在线编码器（query encoder） ---
        self.encoder = _build_temporal_cnn(input_dim, embed_dim)
        self.projection = _build_projection_head(embed_dim)

        # --- 动量编码器（key encoder），参数不参与梯度 ---
        self.momentum_encoder = _build_temporal_cnn(input_dim, embed_dim)
        self.momentum_projection = _build_projection_head(embed_dim)

        # 初始化动量编码器权重 = 在线编码器
        self._init_momentum_encoder()

        # --- 负样本队列（Algorithm 1 line 9） ---
        # 初始化为单位球面上的随机向量（归一化）
        queue_init = F.normalize(torch.randn(embed_dim, queue_size), dim=0)
        self.register_buffer("queue", queue_init)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    # -----------------------------------------------------------------------
    # 初始化
    # -----------------------------------------------------------------------
    def _init_momentum_encoder(self):
        for p_q, p_k in zip(
            list(self.encoder.parameters()) + list(self.projection.parameters()),
            list(self.momentum_encoder.parameters()) + list(self.momentum_projection.parameters()),
        ):
            p_k.data.copy_(p_q.data)
            p_k.requires_grad = False

    # -----------------------------------------------------------------------
    # EMA 更新（Algorithm 1 line 8）
    # θ̄ ← m·θ̄ + (1-m)·θ
    # -----------------------------------------------------------------------
    @torch.no_grad()
    def _update_momentum_encoder(self):
        m = self.momentum
        params_q = list(self.encoder.parameters()) + list(self.projection.parameters())
        params_k = list(self.momentum_encoder.parameters()) + list(self.momentum_projection.parameters())
        for p_q, p_k in zip(params_q, params_k):
            p_k.data.mul_(m).add_((1.0 - m) * p_q.data)

    # -----------------------------------------------------------------------
    # 队列维护（Algorithm 1 line 9）
    # -----------------------------------------------------------------------
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """
        将 keys（B, m）加入负样本队列（循环覆盖）。
        keys 已 L2 归一化。
        """
        B = keys.shape[0]
        ptr = int(self.queue_ptr)
        Q = self.queue_size

        # 处理越界（循环队列）
        if ptr + B <= Q:
            self.queue[:, ptr:ptr + B] = keys.T
        else:
            tail = Q - ptr
            self.queue[:, ptr:] = keys[:tail].T
            self.queue[:, :B - tail] = keys[tail:].T

        self.queue_ptr[0] = (ptr + B) % Q

    # -----------------------------------------------------------------------
    # 编码接口
    # -----------------------------------------------------------------------
    def _encode_online(self, x: torch.Tensor) -> torch.Tensor:
        """
        在线编码器前向传播。
        x: (B, T, D)  →  z: (B, m)，L2 归一化
        """
        feat = self.encoder(x.transpose(1, 2))   # (B, T, D) → (B, D, T) → (B, 512)
        z = self.projection(feat)                 # (B, m)
        return F.normalize(z, dim=1)

    @torch.no_grad()
    def _encode_momentum(self, x: torch.Tensor) -> torch.Tensor:
        """
        动量编码器前向传播（无梯度）。
        x: (B, T, D)  →  z: (B, m)，L2 归一化
        """
        feat = self.momentum_encoder(x.transpose(1, 2))
        z = self.momentum_projection(feat)
        return F.normalize(z, dim=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        推理接口：返回 L2 归一化嵌入，用于 HNSW 检索。
        x: (B, T, D)  →  z: (B, m)
        """
        return self._encode_online(x)

    # -----------------------------------------------------------------------
    # InfoNCE 损失（Eq.3）
    #
    # L_NCE = -E[ log(
    #     exp(q·k⁺/τ_c)
    #     / (exp(q·k⁺/τ_c) + Σ_{i=1}^{M} exp(q·k_i/τ_c))
    # )]
    #
    # 其中：
    #   q = φ_θ(X_anchor)       在线编码器
    #   k⁺ = φ_θ̄(X_positive)    动量编码器
    #   k_i ∈ Q（负样本队列）
    # -----------------------------------------------------------------------
    def contrastive_loss(
        self,
        x_anchor: torch.Tensor,    # (B, T, D)
        x_positive: torch.Tensor,  # (B, T, D)
    ) -> torch.Tensor:
        """
        计算 InfoNCE 损失，同时更新动量编码器和负样本队列。

        返回：标量损失
        """
        # Step 1：在线编码器编码锚点
        q = self._encode_online(x_anchor)   # (B, m)

        # Step 2：动量编码器编码正样本（Algorithm 1 line 8 先更新再编码）
        self._update_momentum_encoder()
        with torch.no_grad():
            k_pos = self._encode_momentum(x_positive)  # (B, m)

        # Step 3：正样本相似度
        l_pos = torch.einsum("bm,bm->b", q, k_pos).unsqueeze(1)  # (B, 1)

        # Step 4：负样本相似度（队列克隆防止梯度流入）
        l_neg = torch.einsum("bm,mk->bk", q, self.queue.clone().detach())  # (B, M)

        # Step 5：拼接 → [正样本, 负样本]，label=0
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature  # (B, 1+M)
        labels = torch.zeros(q.shape[0], dtype=torch.long, device=q.device)

        loss = F.cross_entropy(logits, labels)

        # Step 6：将正样本 key 加入队列（Algorithm 1 line 9）
        self._dequeue_and_enqueue(k_pos)

        return loss
