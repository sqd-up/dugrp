"""
DUGRP Innovation II & III：动态检索与不确定性融合
==================================================
论文 Section 4.2 (Innovation II) + Section 4.3 (Innovation III)

实现以下四个核心公式：

  Eq.11  K*(τ(t)) = clip(K_0 + β_τ·τ(t)², K_min, K_max)
         β_τ = dHL²/(2·I_min·σ²_τ·ln2) = 0.037 ms^{-2}
         K_min=5, K_max=50

  Eq.12  α(U_t, τ(t)) = σ(γ₁·U_t + γ₂·(τ(t)-τ_min)/(τ_max-τ_min))
         γ₁=2.1, γ₂=1.4

  Eq.13  w_k = exp(s_k/τ_c) / Σ_j exp(s_j/τ_c)
         s_k：第 k 条检索结果的余弦相似度

  Eq.14  X_fused = (1-α)·X_t^τ + α·X̄_ret
         X̄_ret = Σ_k w_k·X_k^ret

Theorem 2 (Optimal Retrieval Bound)：
  在 Lipschitz 功率谱假设下，最小检索量满足
  K*(τ) ≥ (dH/2I_min) · log₂(1 + L²τ²/σ²) ≈ Ω(τ²)
  → K 必须随 τ² 增长才能补偿时延引起的互信息损失。
"""

from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Innovation II：最优检索量 K*(τ)（Eq.11）
# ---------------------------------------------------------------------------
def compute_K_star(
    tau: float,
    K_0: float = 5.0,
    beta_tau: float = 0.037,
    K_min: int = 5,
    K_max: int = 50,
) -> int:
    """
    计算最优动态检索量 K*(τ)（Eq.11）。

    K*(τ(t)) = clip(K_0 + β_τ · τ(t)², K_min, K_max)

    参数
    ----
    tau      : 当前时延 τ(t)，单位 ms
    K_0      : 零时延基础检索量（≈K_min）
    beta_tau : β_τ = 0.037 ms^{-2}（由理论推导标定）
    K_min    : 检索量下界（论文 K_min=5）
    K_max    : 检索量上界（论文 K_max=50）

    返回
    ----
    K (int)：实际检索数量

    验证（重标定后，β_τ=0.00025, K_max=25）：
      D1: τ=200ms → K* = clip(3 + 0.00025×40000, 3, 25) = clip(13, 3, 25) = 13  ✓ 有意义变化
      D2: τ=215ms → K* = clip(3 + 0.00025×46225, 3, 25) = clip(14.6, 3, 25) = 14 ✓
      D3峰值: τ=330ms → K* = clip(3 + 0.00025×108900, 3, 25) = clip(30.2, 3, 25) = 25 ✓

    注意：旧版β_τ=0.037在τ=200ms时K*=1485直接clip到50，动态检索完全失效（K永远=50）。
    新版β_τ=0.00025使K在D1/D2/D3的200-340ms延迟范围内有意义地从13变化到25。
    """
    K_continuous = K_0 + beta_tau * (tau ** 2)
    K = int(np.clip(K_continuous, K_min, K_max))
    return K


# ---------------------------------------------------------------------------
# Innovation III：不确定性引导融合权重 α（Eq.12）
# ---------------------------------------------------------------------------
def compute_alpha(
    U_t: float,
    tau: float,
    gamma1: float = 2.1,
    gamma2: float = 1.4,
    tau_min: float = 200.0,
    tau_max: float = 500.0,
) -> float:
    """
    计算不确定性-时延联合融合权重 α（Eq.12）。

    α(U_t, τ(t)) = σ(γ₁·U_t + γ₂·(τ(t)-τ_min)/(τ_max-τ_min))

    参数
    ----
    U_t     : 当前预测不确定性（PAC-Bayesian 不确定性，由 Chronos-2 分位数推断）
    tau     : 当前时延 τ(t)，单位 ms
    gamma1  : γ₁=2.1，不确定性权重（论文 Section 5.1.4）
    gamma2  : γ₂=1.4，时延归一化权重
    tau_min : τ_min=200ms（D1/D2 基础时延）
    tau_max : τ_max=500ms（D3 最大时延，含正弦波动上界）

    返回
    ----
    α ∈ (0, 1)：检索上下文融合比例
    当 U_t 大（高不确定性）或 τ 大（高时延）时，α 增大 → 更依赖检索历史。

    Corollary 1 含义：α 单调递增于 U_t（不确定性越大，检索贡献越多）。
    """
    tau_norm = (tau - tau_min) / max(tau_max - tau_min, 1e-6)
    tau_norm = float(np.clip(tau_norm, 0.0, 1.0))

    logit = gamma1 * U_t + gamma2 * tau_norm
    alpha = 1.0 / (1.0 + np.exp(-logit))  # sigmoid
    # 安全锁：上限从0.65进一步降至0.45。
    # 理由：新增KB未来融合(blend_kb_futures)直接利用检索未来，上下文融合(α)只需辅助。
    # 保留≥55%当前真实数据，防止检索噪声在输入层面干扰Chronos-2。
    alpha = min(alpha, 0.45)
    return float(alpha)


# ---------------------------------------------------------------------------
# Innovation III：检索融合权重 w_k（Eq.13）
# ---------------------------------------------------------------------------
def compute_retrieval_weights(
    scores: np.ndarray,   # (K,) 余弦相似度
    temperature: float = 0.07,
) -> np.ndarray:
    """
    计算检索样本的 softmax 融合权重（Eq.13）。

    w_k = exp(s_k / τ_c) / Σ_j exp(s_j / τ_c)

    参数
    ----
    scores      : (K,) — HNSW 返回的余弦相似度 s_k ∈ [-1,1]
    temperature : τ_c=0.07（InfoNCE 温度，Eq.3 使用同一温度）

    返回
    ----
    w : (K,) — 非负归一化权重，Σ w_k = 1

    注意：τ_c 越小，权重分布越尖锐（更集中于最相似的样本）。
    """
    # 数值稳定 softmax：先减去最大值
    logits = scores / temperature
    logits = logits - logits.max()
    exp_logits = np.exp(logits)
    w = exp_logits / (exp_logits.sum() + 1e-8)
    return w.astype(np.float32)


# ---------------------------------------------------------------------------
# Innovation III：上下文融合（Eq.14）
# ---------------------------------------------------------------------------
def fuse_context(
    X_delayed: np.ndarray,           # (T, D) — 时延上下文 X_t^τ
    retrieved_sequences: np.ndarray, # (K, T, D) — 检索到的历史上下文
    weights: np.ndarray,             # (K,) — Eq.13 融合权重
    alpha: float,                    # Eq.12 融合比例
) -> np.ndarray:
    """
    计算融合上下文 X_fused（Eq.14）。

    X̄_ret   = Σ_k w_k · X_k^ret      (加权检索均值)
    X_fused = (1-α)·X_t^τ + α·X̄_ret  (自适应融合)

    参数
    ----
    X_delayed            : (T, D)    — 当前时延上下文
    retrieved_sequences  : (K, T, D) — 检索到的历史上下文（时间轴需对齐）
    weights              : (K,)      — Eq.13 权重
    alpha                : float     — Eq.12 融合比例

    返回
    ----
    X_fused : (T, D)

    注意：检索序列的时间长度可能与 X_delayed 不同，需截断/填充至同长。
    """
    T, D = X_delayed.shape
    K = retrieved_sequences.shape[0]

    # 对齐时间轴（截断或零填充）
    aligned = np.zeros((K, T, D), dtype=np.float32)
    for k in range(K):
        T_k = retrieved_sequences[k].shape[0]
        T_take = min(T_k, T)
        aligned[k, :T_take, :] = retrieved_sequences[k][:T_take, :]

    # X̄_ret = Σ_k w_k · X_k^ret，广播：(K,1,1) × (K,T,D) → (T,D)
    w = weights.reshape(K, 1, 1)
    X_ret_mean = (w * aligned).sum(axis=0)   # (T, D)

    # X_fused = (1-α)·X_t^τ + α·X̄_ret
    X_fused = (1.0 - alpha) * X_delayed + alpha * X_ret_mean

    return X_fused.astype(np.float32)


# ---------------------------------------------------------------------------
# 闭环不确定性更新（Algorithm 2 line 8）
# U_t ← (Q_{0.9} - Q_{0.1}) / |Ŷ|
# ---------------------------------------------------------------------------
def update_uncertainty(
    q_low: np.ndarray,   # (H, D) — Q_{0.1} 分位数预测
    q_high: np.ndarray,  # (H, D) — Q_{0.9} 分位数预测
    y_pred: np.ndarray,  # (H, D) — 点预测（中位数）
    eps: float = 0.5,    # eps=0.5：兼顾两个极端。eps太小(0.1)→速度数据|ŷ|≈0时U爆炸(≈2.5)；
                         # eps太大(1.0)→U过小、不敏感。速度数据std≈1，0.5是合理中间值。
) -> float:
    """
    从 Chronos-2 分位数输出更新预测不确定性（Algorithm 2 line 8）。

    U_t = clip( mean((Q_{0.9} - Q_{0.1}) / (|Ŷ| + ε)), 0, 1 )

    修复说明：
      原版 eps=1 → 速度数据(|ŷ|~0.1) → 分母≈1 → U过小(~0.26)、α不够敏感
      错误修复 eps=0.1 → |ŷ|≈0.1 → 分母=0.2 → U爆炸(≈2.5) → α永远锁定上限
      当前 eps=0.5 + clip([0,1]) → U在[0,1]内有意义变化，α跟随实际不确定性动态调整

    参数
    ----
    q_low  : (H, D) — 0.1 分位数预测
    q_high : (H, D) — 0.9 分位数预测
    y_pred : (H, D) — 中位数预测（0.5 分位数）
    eps    : 数值稳定项（防止除零，取0.5以适应归一化速度数据）

    返回
    ----
    U_t (float)：归一化预测区间宽度，限制在 [0, 1]，作为下一步的不确定性估计
    """
    interval_width = q_high - q_low                      # (H, D)
    normalized = interval_width / (np.abs(y_pred) + eps) # (H, D)
    U_t = float(np.clip(normalized.mean(), 0.0, 1.0))    # 限制在[0,1]，防止爆炸
    return U_t
