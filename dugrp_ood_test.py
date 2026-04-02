# -*- coding: utf-8 -*-
import os
import sys
import json
import torch
import numpy as np

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

# 将当前目录加入路径，以保证能正确 import dugrp
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from dugrp.config import DUGRPConfig
from dugrp.predictor import DUGRPPredictor
from dugrp.evaluate import compute_mae, compute_rmse, compute_crps
from dugrp.datasets import inject_delay, DELAY_PROFILES

def evaluate_dugrp_sequences(predictor, sequences, val_end, L, H, scenario_name="Standard"):
    print(f"\n--- Starting Evaluation: {scenario_name} Environment ---")
    results = {}

    for profile_name in ["D1", "D2", "D3"]:
        profile = DELAY_PROFILES[profile_name]
        
        mae_list, rmse_list, crps_list = [], [], []
        
        # 每次评估一个新的场景前，重置闭环不确定性 U_t
        predictor.reset_uncertainty()
        
        for seq in sequences[val_end:]:
            delayed, delays = inject_delay(seq, profile, dt_ms=100.0)
            
            # 步长为 5 的滑动窗口（与 Baseline 绝对一致）
            for t in range(L, len(seq) - H + 1, 5):
                ctx = delayed[t - L : t]
                fut = seq[t : t + H]
                # 当前观测时刻的真实延迟量，提供给 K* 和 α 的计算
                tau = float(delays[t - 1]) 
                
                # DUGRP 核心预测 (包含动态检索 + 融合 + Chronos-2 推理)
                res = predictor.predict(context=ctx, tau=tau, use_retrieval=True)
                
                mae_list.append(compute_mae(res.y_pred, fut))
                rmse_list.append(compute_rmse(res.y_pred, fut))
                crps_list.append(compute_crps(res.q_low, res.q_high, fut))
                
        profile_res = {
            "MAE": float(np.mean(mae_list)),
            "RMSE": float(np.mean(rmse_list)),
            "CRPS": float(np.mean(crps_list))
        }
        results[profile_name] = profile_res
        print(f"[{scenario_name} - {profile_name}] MAE: {profile_res['MAE']:.4f} | RMSE: {profile_res['RMSE']:.4f} | CRPS: {profile_res['CRPS']:.4f}")
    
    return results

def run_dugrp_ood_experiment():
    print("=" * 60)
    print("Running DUGRP Final Test: Standard vs OOD (Gaussian Turbulence)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = "data/rov_data.npy"
    checkpoint_dir = "checkpoints/rov"  # 指向你已经训练好的模型和 KB

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # 1. 加载数据（必须和 baseline 一样做差分和归一化）
    sequences_raw = np.load(data_path)
    sequences = np.diff(sequences_raw, axis=1)
    
    N_total = len(sequences)
    train_end = int(N_total * 0.7)
    val_end   = int(N_total * 0.8)

    train_data_raw = sequences[:train_end]
    mean = train_data_raw.mean(axis=(0, 1), keepdims=True)
    std  = train_data_raw.std(axis=(0, 1), keepdims=True) + 1e-8
    
    sequences = (sequences - mean) / std

    # 2. 初始化 DUGRP 并加载你的离线模型与 KB
    config = DUGRPConfig()
    config.chronos_model_id = "/home/c201/sqd/models/chronos-2" # 本地绝对路径
    
    print(f"Loading DUGRP from {checkpoint_dir}...")
    predictor = DUGRPPredictor(config)
    predictor.load(checkpoint_dir)
    
    # 预热 Chronos-2，防止初次调用耗时过长
    predictor.warmup(n_calls=5)

    L = config.context_length
    H = config.prediction_length

    # 3. Evaluate on Standard Environment
    standard_results = evaluate_dugrp_sequences(
        predictor, sequences, val_end, L, H, scenario_name="Standard"
    )

    # 4. Apply OOD Test: Simulated Ocean Turbulence (高频高斯乱流)
    print("\n" + "=" * 60)
    print("Applying OOD Test: Simulated Ocean Turbulence (Gaussian Noise)")
    print("=" * 60)
    
    sequences_ood = sequences.copy()
    
    # 注入标准差为 0.5 的高频高斯白噪声，精确对齐 Vanilla Chronos-2 的 OOD 测试
    np.random.seed(42) # 固定种子，保证扰动形态一致
    noise_intensity = 0.2
    turbulence = np.random.normal(0, noise_intensity, size=sequences_ood[val_end:].shape)
    sequences_ood[val_end:] += turbulence
    
    ood_results = evaluate_dugrp_sequences(
        predictor, sequences_ood, val_end, L, H, scenario_name="OOD_Turbulence"
    )

    # 5. 保存结果
    final_results = {
        "Standard_Environment": standard_results,
        "OOD_Turbulence_Environment": ood_results
    }
    
    res_file = "dugrp_rov_results_with_ood.json"
    with open(res_file, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nAll tests completed! Results saved to {res_file}")

if __name__ == "__main__":
    run_dugrp_ood_experiment()
