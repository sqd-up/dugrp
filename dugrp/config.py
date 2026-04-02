"""
DUGRP 全局配置
==============
所有超参数严格对应论文各节，禁止在其他文件中硬编码数值。

论文对照：
  - Section 5.1.4 (Implementation details)
  - Table 1 (Delay profiles)
  - Table 4 (HNSW latency & recall)
  - Algorithm 1 & 2
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


@dataclass
class DUGRPConfig:
    # -----------------------------------------------------------------------
    # 数据维度
    # -----------------------------------------------------------------------
    input_dim: int = 6           # D：控制信号维度（6-DOF BlueROV2）
    context_length: int = 64     # L：上下文窗口长度（64=~6.4s@10Hz，T5编码器O(L²)→4x提速）
    prediction_length: int = 24  # H：预测步长
    use_first_diff: bool = True  # 是否对ROV位置数据做一阶差分（转为速度，消除非平稳性）

    # -----------------------------------------------------------------------
    # AR-Chronos Horizon-Adaptive 融合
    # AR (线性自回归) 对短期速度预测天然精准；Chronos-2 擅长长期非线性模式。
    # 指数衰减权重：w_ar(h) = exp(-h / λ)，AR 主导近期，Chronos-2 主导远期。
    # -----------------------------------------------------------------------
    use_ar_blend: bool = True        # 是否启用 AR-Chronos 自适应融合
    ar_blend_horizon: float = 6.0    # λ：AR权重半衰期（步数），越小AR作用越短

    # -----------------------------------------------------------------------
    # KB未来融合（Innovation IV：RAG-for-TimeSeries）
    # 直接用检索到的历史未来值增强预测。
    # 修复：移除硬阈值(threshold=0.3曾导致编码器未充分训练时blend完全失效)
    # 新公式：blend_coef = clip(mean_score, 0, 1) × kb_blend_max（始终激活）
    # -----------------------------------------------------------------------
    blend_kb_futures: bool = True    # 是否启用KB未来融合
    kb_blend_max: float = 0.60       # 最大融合系数（mean_score=1时生效）

    # -----------------------------------------------------------------------
    # 对比学习编码器（Section 4.1, Eq.3）
    # "Contrastive encoder: 4-layer temporal CNN, m=128"
    # -----------------------------------------------------------------------
    embed_dim: int = 128         # m：球面嵌入维度 S^{m-1}
    temperature: float = 0.07    # τ_c：InfoNCE 温度（标准 MoCo 值）
    queue_size: int = 65536      # M：负样本队列大小
    momentum: float = 0.99       # EMA 动量系数（论文 Algorithm 1 line 8）

    # -----------------------------------------------------------------------
    # HNSW 索引（Section 4.1.3, Table 4）
    # "ef_search=50 → recall@10=0.997, latency 2.13ms at N=10^5"
    # -----------------------------------------------------------------------
    hnsw_M: int = 32             # HNSW 每层邻居数（FAISS 默认推荐值）
    hnsw_ef_construction: int = 200  # 构建时的搜索宽度
    hnsw_ef_search: int = 50     # ef_J=50（Table 4）

    # -----------------------------------------------------------------------
    # 动态检索（Theorem 2, Eq.11）
    # K*(τ(t)) = clip(K_0 + β_τ · τ(t)², K_min, K_max)
    # β_τ重标定：原值0.037在τ=200ms时K*=1485直接saturate至50，动态检索完全失效。
    # 新值0.00025使K*(200ms)≈13, K*(300ms)≈25.5→25，在D1/D2/D3范围内有意义变化。
    # -----------------------------------------------------------------------
    K_0: float = 3.0             # 零时延基础检索量
    K_min: int = 3               # 检索量下界
    K_max: int = 25              # 检索量上界（与β_τ重标定值配套）
    beta_tau: float = 0.00025    # β_τ(ms^{-2})重标定：K*(200ms)≈13, K*(300ms)≈25

    # -----------------------------------------------------------------------
    # 不确定性引导融合（Eq.12）
    # α(U_t, τ(t)) = σ(γ₁·U_t + γ₂·(τ(t)-τ_min)/(τ_max-τ_min))
    # 调低γ₁/γ₂：原值使α易达到0.636+，KB质量未充分训练时过度依赖检索会增大误差。
    # -----------------------------------------------------------------------
    gamma1: float = 1.5          # γ₁：不确定性权重（从2.1降低）
    gamma2: float = 1.0          # γ₂：时延归一化权重（从1.4降低）
    tau_min: float = 200.0       # τ_min (ms)，对应 D1/D2 基础时延
    tau_max: float = 500.0       # τ_max (ms)，对应 D3 最大时延

    # -----------------------------------------------------------------------
    # Chronos-2（Section 4.4）
    # "Employing Chronos-2 [12], 46M parameters, quantized tokenization"
    # -----------------------------------------------------------------------
    chronos_model_id: str = "/home/c201/sqd/models/chronos-2"
    quantile_levels: List[float] = field(
        default_factory=lambda: [0.1, 0.5, 0.9]
    )

    # -----------------------------------------------------------------------
    # 训练（Section 5）
    # Phase 1: 对比预训练（Algorithm 1）
    # Phase 2: 在线微调（Algorithm 2 闭环）
    # -----------------------------------------------------------------------
    learning_rate: float = 3e-4     # 提高lr：加速对比学习收敛（从1e-4提升）
    weight_decay: float = 1e-5
    batch_size: int = 64            # 增大batch：MoCo负样本更丰富（从32提升）
    epochs_contrastive: int = 150   # Phase 1 离线对比训练轮数（从100提升）
    epochs_finetune: int = 50       # Phase 2 微调轮数
    neighbor_radius: float = 0.07   # r：正样本邻域半径（余弦距离，Assumption 1）
    neighbor_window: int = 30       # 时序正样本采样窗口（±步数）

    # -----------------------------------------------------------------------
    # 运行
    # -----------------------------------------------------------------------
    device: str = "cuda"
    seed: int = 42
    num_workers: int = 4