"""
DUGRP 评估模块
==============
论文 Section 5.1.3 (Metrics) + Table 2/3/5/6

实现以下评估指标：
  - MAE：平均绝对误差（主要指标，论文目标 0.088）
  - RMSE：均方根误差
  - CRPS：连续排名概率分数（衡量预测分布质量）
  - KHR@K：知识命中率（Knowledge Hit Rate，Table 6）
             KHR@K = fraction of retrieved samples within ε-MAE of true future

消融变体（Table 5）：
  DUGRP-noKB   (use_retrieval=False)
  DUGRP-staticK (fixed K=10)
  DUGRP-noUQ   (fixed α=1, 不依赖不确定性)
  DUGRP-noClosed (禁止闭环 U_t 更新)
  DUGRP-noC2   (线性预测替代 Chronos-2)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import DUGRPConfig
from .predictor import DUGRPPredictor, PredictionResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 单指标计算函数
# ---------------------------------------------------------------------------
def compute_mae(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """MAE = mean|ŷ - y|"""
    return float(np.abs(y_pred - y_true).mean())


def compute_rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """RMSE = sqrt(mean(ŷ - y)²)"""
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def compute_crps(
    q_low: np.ndarray,    # (H, D) Q_{0.1}
    q_high: np.ndarray,   # (H, D) Q_{0.9}
    y_true: np.ndarray,   # (H, D)
) -> float:
    """
    CRPS（连续排名概率分数）的近似估计（Section 5.1.3）。

    论文使用 CRPS probabilistic，此处用 interval score 近似：
    CRPS ≈ (q_high - q_low) + (2/0.8)·max(q_low - y, 0) + (2/0.8)·max(y - q_high, 0)

    = 区间宽度 + 区间外惩罚（α=0.8 对应 80% 预测区间）
    """
    alpha = 0.8   # 预测区间置信度 (90% - 10% = 80%)
    width = q_high - q_low
    low_viol  = np.maximum(q_low  - y_true, 0)
    high_viol = np.maximum(y_true - q_high, 0)
    crps = width + (2 / alpha) * low_viol + (2 / alpha) * high_viol
    return float(crps.mean())


def compute_khr(
    predictor: DUGRPPredictor,
    query_contexts: np.ndarray,   # (N, T, D)
    true_futures: np.ndarray,     # (N, H, D)
    K: int = 10,
    epsilon: Optional[float] = None,
) -> float:
    """
    KHR@K（知识命中率，Table 6）。

    KHR@K = (1/N) Σ_i 1[∃k ∈ top-K : MAE(X_k^fut, Y_i^true) ≤ ε]

    其中 ε = MAE(全局均值预测, y_true)（自适应阈值，论文未明确指定 ε 定义，
    此处按常见 RAG 评估方式实现）。

    参数
    ----
    K       : 检索数量（Table 6 给出 KHR@5, KHR@10, KHR@20）
    epsilon : 命中阈值（None → 使用 MAE 全局均值的 1.0 倍）
    """
    if predictor.kb.size == 0:
        logger.warning("[KHR] 知识库为空，KHR=0")
        return 0.0

    from .dynamic_retrieval import compute_K_star
    cfg = predictor.cfg

    # 计算自适应 ε
    if epsilon is None:
        global_mean = true_futures.mean(axis=(1, 2))  # (N,)
        global_pred = np.full_like(true_futures, true_futures.mean())
        epsilon = float(np.abs(global_pred - true_futures).mean())

    hits = 0
    N = len(query_contexts)

    for i in range(N):
        emb = predictor._embed(query_contexts[i])
        result = predictor.kb.search(emb, K)

        # 检查是否有检索结果命中真值 future（在 ε-MAE 内）
        hit = False
        for k_idx in range(len(result.futures)):
            H_r = result.futures[k_idx].shape[0]
            H_q = true_futures[i].shape[0]
            H_min = min(H_r, H_q)
            mae = float(np.abs(result.futures[k_idx][:H_min] - true_futures[i][:H_min]).mean())
            if mae <= epsilon:
                hit = True
                break
        hits += int(hit)

    return hits / N


# ---------------------------------------------------------------------------
# 评估结果数据结构
# ---------------------------------------------------------------------------
@dataclass
class EvaluationResult:
    """单场景评估结果（对应论文 Table 2/3）。"""
    delay_profile: str
    mae:    float
    rmse:   float
    crps:   float
    mean_latency_ms:  float
    std_latency_ms:   float
    n_samples: int
    # 额外调试信息
    mean_K:     float = 0.0   # 平均检索量
    mean_alpha: float = 0.0   # 平均融合权重
    mean_U:     float = 0.0   # 平均不确定性


# ---------------------------------------------------------------------------
# 主评估器
# ---------------------------------------------------------------------------
class DUGRPEvaluator:
    """
    DUGRP 完整评估器。

    用法：
      evaluator = DUGRPEvaluator(predictor)
      results = evaluator.evaluate_all(
          contexts, futures, delays,
          delay_profile_name="D3"
      )
    """

    def __init__(self, predictor: DUGRPPredictor):
        self.predictor = predictor

    def evaluate_all(
        self,
        contexts: np.ndarray,           # (N, L, D)
        futures:  np.ndarray,           # (N, H, D)
        delays:   np.ndarray,           # (N,) — 每步实际时延 (ms)
        delay_profile_name: str = "D3",
        use_retrieval: bool = True,
    ) -> EvaluationResult:
        """
        对给定测试集执行完整评估（Algorithm 2 逐步调用）。

        参数
        ----
        contexts  : (N, L, D) — 时延上下文
        futures   : (N, H, D) — 真值未来序列
        delays    : (N,)      — 每步时延 (ms)
        delay_profile_name : 时延场景名称（用于报告）
        use_retrieval : 是否启用检索（消融实验）

        返回
        ----
        EvaluationResult
        """
        N = len(contexts)
        assert len(futures) == N and len(delays) == N

        # 重置闭环状态（每次独立评估）
        self.predictor.reset_uncertainty()

        mae_list:    List[float] = []
        rmse_list:   List[float] = []
        crps_list:   List[float] = []
        latency_list: List[float] = []
        K_list:     List[float] = []
        alpha_list: List[float] = []
        U_list:     List[float] = []

        for i in range(N):
            t0 = time.perf_counter()
            result: PredictionResult = self.predictor.predict(
                context=contexts[i],
                tau=float(delays[i]),
                use_retrieval=use_retrieval,
            )
            latency_ms = (time.perf_counter() - t0) * 1000

            # 对齐预测步长与真值步长
            H_pred = result.y_pred.shape[0]
            H_true = futures[i].shape[0]
            H_min  = min(H_pred, H_true)

            y_pred = result.y_pred[:H_min]
            y_true = futures[i][:H_min]
            q_low  = result.q_low[:H_min]
            q_high = result.q_high[:H_min]

            mae_list.append(compute_mae(y_pred, y_true))
            rmse_list.append(compute_rmse(y_pred, y_true))
            crps_list.append(compute_crps(q_low, q_high, y_true))
            latency_list.append(latency_ms)
            K_list.append(float(result.K))
            alpha_list.append(float(result.alpha))
            U_list.append(float(result.uncertainty))

            if (i + 1) % 100 == 0:
                logger.info(
                    f"[Eval] {delay_profile_name} | {i+1}/{N} | "
                    f"MAE={np.mean(mae_list):.4f} | "
                    f"latency={np.mean(latency_list):.1f}ms | "
                    f"K={np.mean(K_list):.1f}"
                )

        return EvaluationResult(
            delay_profile=delay_profile_name,
            mae=float(np.mean(mae_list)),
            rmse=float(np.mean(rmse_list)),
            crps=float(np.mean(crps_list)),
            mean_latency_ms=float(np.mean(latency_list)),
            std_latency_ms=float(np.std(latency_list)),
            n_samples=N,
            mean_K=float(np.mean(K_list)),
            mean_alpha=float(np.mean(alpha_list)),
            mean_U=float(np.mean(U_list)),
        )

    def ablation_study(
        self,
        contexts: np.ndarray,
        futures:  np.ndarray,
        delays:   np.ndarray,
    ) -> Dict[str, EvaluationResult]:
        """
        执行论文 Table 5 消融实验。

        消融变体：
          DUGRP (full)       — 完整模型
          DUGRP-noKB         — 无知识库检索
          DUGRP-staticK      — 固定 K=10
          DUGRP-noUQ         — 固定 α=1（不依赖不确定性）
          DUGRP-noClosed     — 禁止 U_t 闭环更新

        返回 {variant_name: EvaluationResult}
        """
        cfg = self.predictor.cfg
        results: Dict[str, EvaluationResult] = {}

        # 1. DUGRP (full)
        logger.info("[Ablation] 评估 DUGRP (full)...")
        results["DUGRP (full)"] = self.evaluate_all(contexts, futures, delays, "D3", True)

        # 2. DUGRP-noKB（无检索，对应 Table 5 "+14.8% MAE"）
        logger.info("[Ablation] 评估 DUGRP-noKB...")
        results["DUGRP-noKB"] = self.evaluate_all(contexts, futures, delays, "D3", False)

        # 3. DUGRP-staticK（固定 K=10，对应 Table 5 "+20.5% MAE"）
        logger.info("[Ablation] 评估 DUGRP-staticK (K=10)...")
        orig_K_max = cfg.K_max
        orig_K_min = cfg.K_min
        orig_beta  = cfg.beta_tau
        cfg.K_min = 10; cfg.K_max = 10; cfg.beta_tau = 0.0   # 固定 K=10
        results["DUGRP-staticK"] = self.evaluate_all(contexts, futures, delays, "D3", True)
        cfg.K_min = orig_K_min; cfg.K_max = orig_K_max; cfg.beta_tau = orig_beta

        # 4. DUGRP-noUQ（α 固定为常数 0.5，不依赖 U_t）
        logger.info("[Ablation] 评估 DUGRP-noUQ...")
        orig_gamma1 = cfg.gamma1
        cfg.gamma1 = 0.0   # α = σ(γ₂·τ_norm)，去除不确定性项
        results["DUGRP-noUQ"] = self.evaluate_all(contexts, futures, delays, "D3", True)
        cfg.gamma1 = orig_gamma1

        # 5. DUGRP-noClosed（禁止更新 U_t，固定为初始值）
        logger.info("[Ablation] 评估 DUGRP-noClosed...")
        self.predictor.reset_uncertainty(U_init=0.5)
        orig_update = self.predictor._U_prev

        class _NoClosed:
            """临时 patch：predict 后不更新 _U_prev"""
            pass

        # 保存原始 predict 方法，临时重写不确定性更新
        original_predict = self.predictor.predict.__func__

        def predict_no_closed(self_pred, context, tau, use_retrieval=True):
            result = original_predict(self_pred, context, tau, use_retrieval)
            self_pred._U_prev = 0.5   # 强制固定，不更新
            return result

        import types
        self.predictor.predict = types.MethodType(predict_no_closed, self.predictor)
        results["DUGRP-noClosed"] = self.evaluate_all(contexts, futures, delays, "D3", True)
        # 恢复原始方法
        self.predictor.predict = types.MethodType(original_predict, self.predictor)

        return results

    def knowledge_hit_rate(
        self,
        contexts: np.ndarray,
        futures:  np.ndarray,
        K_values: List[int] = [5, 10, 20],
    ) -> Dict[str, float]:
        """
        计算 KHR@K（Table 6）。

        返回 {"KHR@5": ..., "KHR@10": ..., "KHR@20": ...}
        """
        result = {}
        for K in K_values:
            khr = compute_khr(self.predictor, contexts, futures, K=K)
            result[f"KHR@{K}"] = khr
            logger.info(f"[KHR] KHR@{K} = {khr:.3f}")
        return result

    def print_summary(self, result: EvaluationResult):
        """打印评估结果摘要（对比论文 Table 2）。"""
        print(f"\n{'='*60}")
        print(f"DUGRP 评估结果 — 时延场景: {result.delay_profile}")
        print(f"{'='*60}")
        print(f"  MAE    : {result.mae:.4f}   (论文目标: 0.088)")
        print(f"  RMSE   : {result.rmse:.4f}  (论文目标: 0.121)")
        print(f"  CRPS   : {result.crps:.4f}  (论文目标: 0.099)")
        print(f"  延迟   : {result.mean_latency_ms:.1f}±{result.std_latency_ms:.1f}ms"
              f"  (论文目标: ≤12ms)")
        print(f"  样本数 : {result.n_samples}")
        print(f"  平均K  : {result.mean_K:.1f}  (K_min=5, K_max=50)")
        print(f"  平均α  : {result.mean_alpha:.3f}")
        print(f"  平均U  : {result.mean_U:.4f}")
        print(f"{'='*60}\n")
