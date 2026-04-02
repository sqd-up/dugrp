"""
DUGRP: Delay-Aware Dynamic Knowledge Retrieval and
       Uncertainty-Guided Co-Learning for Streaming Time Series Prediction
==========================================================================
IEEE Transactions on Knowledge and Data Engineering (TKDE) 投稿版本

三大理论创新（均有数学证明）：
  Theorem 1：拓扑分离边界 Δ > 0（对比知识库，O(log N) HNSW 检索）
  Theorem 2：最优检索量 K*(τ) = Ω(τ²)（动态检索量，Eq.11）
  Theorem 3：PAC-Bayesian 界收紧（不确定性协同学习，Eq.16）

使用示例：
  from dugrp import DUGRPConfig, DUGRPPredictor, DUGRPTrainer, DUGRPEvaluator

  config    = DUGRPConfig(device="cuda")
  predictor = DUGRPPredictor(config)
  trainer   = DUGRPTrainer(config, predictor)

  # Phase 1+2：训练 + 构建知识库
  trainer.fit(train_sequences)

  # 在线推理（Algorithm 2）
  result = predictor.predict(context, tau=300.0)
  print(result.y_pred)        # (H, D) 点预测
  print(result.uncertainty)   # U_t 不确定性
"""

__version__ = "1.0.0"
__paper__ = "Delay-Aware Dynamic Knowledge Retrieval and Uncertainty-Guided Co-Learning for Streaming Time Series Prediction"
__venue__ = "IEEE TKDE"

# --- 配置 ---
from .config import DUGRPConfig

# --- 核心模块（三大支柱） ---
from .encoder import ContrastiveEncoder
from .knowledge_base import KnowledgeBase, RetrievalResult
from .dynamic_retrieval import (
    compute_K_star,
    compute_alpha,
    compute_retrieval_weights,
    fuse_context,
    update_uncertainty,
)

# --- 预测器（Algorithm 2） ---
from .predictor import DUGRPPredictor, PredictionResult

# --- 训练器（Algorithm 1） ---
from .trainer import DUGRPTrainer, ContrastiveDataset

# --- 数据集 ---
from .datasets import (
    ETTh1Dataset,
    DelayProfileD1,
    DelayProfileD2,
    DelayProfileD3,
    DELAY_PROFILES,
    inject_delay,
    make_synthetic_rov_data,
)

# --- 评估 ---
from .evaluate import (
    DUGRPEvaluator,
    EvaluationResult,
    compute_mae,
    compute_rmse,
    compute_crps,
    compute_khr,
)

__all__ = [
    # Config
    "DUGRPConfig",
    # Core modules
    "ContrastiveEncoder",
    "KnowledgeBase",
    "RetrievalResult",
    "compute_K_star",
    "compute_alpha",
    "compute_retrieval_weights",
    "fuse_context",
    "update_uncertainty",
    # Predictor
    "DUGRPPredictor",
    "PredictionResult",
    # Trainer
    "DUGRPTrainer",
    "ContrastiveDataset",
    # Datasets
    "ETTh1Dataset",
    "DelayProfileD1",
    "DelayProfileD2",
    "DelayProfileD3",
    "DELAY_PROFILES",
    "inject_delay",
    "make_synthetic_rov_data",
    # Evaluator
    "DUGRPEvaluator",
    "EvaluationResult",
    "compute_mae",
    "compute_rmse",
    "compute_crps",
    "compute_khr",
]
