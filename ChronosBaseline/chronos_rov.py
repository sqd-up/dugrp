# -*- coding: utf-8 -*-
import os
import sys
import time
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from chronos.chronos2 import Chronos2Pipeline

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dugrp.datasets import inject_delay, DELAY_PROFILES
from dugrp.evaluate import compute_mae, compute_rmse, compute_crps

def evaluate_chronos_on_sequences(pipeline, sequences, val_end, L, H, batch_size, device, scenario_name="Standard"):
    print(f"\n--- Starting Evaluation: {scenario_name} Environment ---")
    results = {}

    for profile_name in ["D1", "D2", "D3"]:
        profile = DELAY_PROFILES[profile_name]
        ctx_list, fut_list = [], []
        
        for seq in sequences[val_end:]:
            delayed, _ = inject_delay(seq, profile, dt_ms=100.0)
            for t in range(L, len(seq) - H + 1, 5):
                ctx_list.append(delayed[t - L : t])
                fut_list.append(seq[t : t + H])
                
        X_test = torch.tensor(np.array(ctx_list), dtype=torch.float32)
        Y_test = np.array(fut_list).astype(np.float32)
        
        test_dataset = TensorDataset(X_test)
        # batch_size=32 避免爆显存
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        mae_list, rmse_list, crps_list = [], [], []
        
        idx = 0
        use_amp = (device.type == "cuda")
        
        for (bx,) in test_loader:
            actual_batch_size = bx.shape[0]
            
            # 严格对齐 predictor.py 中的数据封装格式：{"target": (D, T)}
            chronos_inputs = []
            for i in range(actual_batch_size):
                target = bx[i].cpu().numpy().T  # 形状转换为 (6, L)
                chronos_inputs.append({"target": target})
            
            # 严格对齐 predictor.py 中的 predict_quantiles 接口
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=use_amp):
                    quantiles_list, mean_list = pipeline.predict_quantiles(
                        chronos_inputs, 
                        prediction_length=H, 
                        quantile_levels=[0.1, 0.5, 0.9]
                    )
            
            batch_y_true = Y_test[idx : idx + actual_batch_size]
            idx += actual_batch_size
            
            for i in range(actual_batch_size):
                q_tensor = quantiles_list[i]
                if torch.is_tensor(q_tensor):
                    q_tensor = q_tensor.cpu().numpy()
                
                # q_tensor 形状为 (D, H, 3)，D=6。我们需要将其转置为 (H, 6) 用于评估
                q_low_i = q_tensor[:, :, 0].T
                mean_pred_i = q_tensor[:, :, 1].T  # 0.5 分位数作为点预测真值
                q_high_i = q_tensor[:, :, 2].T
                
                y_true_i = batch_y_true[i]
                
                mae_list.append(compute_mae(mean_pred_i, y_true_i))
                rmse_list.append(compute_rmse(mean_pred_i, y_true_i))
                crps_list.append(compute_crps(q_low_i, q_high_i, y_true_i))
                    
        res = {
            "MAE": float(np.mean(mae_list)),
            "RMSE": float(np.mean(rmse_list)),
            "CRPS": float(np.mean(crps_list))
        }
        results[profile_name] = res
        print(f"[{scenario_name} - {profile_name}] MAE: {res['MAE']:.4f} | RMSE: {res['RMSE']:.4f} | CRPS: {res['CRPS']:.4f}")
    
    return results

def run_chronos_baseline_experiment():
    print("=" * 60)
    print("Running Baseline: Vanilla Chronos-2 (Direct Quantile Inference)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    L = 64
    H = 24
    data_path = "../data/rov_data.npy"
    chronos_model_id = "/home/c201/sqd/models/chronos-2"

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    sequences_raw = np.load(data_path)
    sequences = np.diff(sequences_raw, axis=1)
    
    N_total = len(sequences)
    train_end = int(N_total * 0.7)
    val_end   = int(N_total * 0.8)

    train_data_raw = sequences[:train_end]
    mean = train_data_raw.mean(axis=(0, 1), keepdims=True)
    std  = train_data_raw.std(axis=(0, 1), keepdims=True) + 1e-8
    
    sequences = (sequences - mean) / std

    print(f"Loading Chronos Pipeline ({chronos_model_id})...")
    try:
        pipeline = Chronos2Pipeline.from_pretrained(
            chronos_model_id,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        )
    except Exception as e:
        print(f"Failed with bfloat16, using float32. Error: {e}")
        pipeline = Chronos2Pipeline.from_pretrained(
            chronos_model_id,
            device_map="cuda",
            torch_dtype=torch.float32,
        )

    # 1. Evaluate on Standard Environment
    standard_results = evaluate_chronos_on_sequences(
        pipeline, sequences, val_end, L, H, batch_size=32, device=device, scenario_name="Standard"
    )

    # 2. Apply OOD Test: Simulated Ocean Turbulence (突发洋流乱流/噪声)
    print("\n" + "=" * 60)
    print("Applying OOD Test: Simulated Ocean Turbulence (Gaussian Noise)")
    print("=" * 60)
    
    sequences_ood = sequences.copy()
    # 注入标准差为 0.5 的高斯白噪声，模拟强烈的洋流乱流和传感器抖动
    # 这种高频、无规律的扰动是 Chronos 无法通过简单的去均值来抵消的
    noise_intensity = 0.2
    turbulence = np.random.normal(0, noise_intensity, size=sequences_ood[val_end:].shape)
    sequences_ood[val_end:] += turbulence
    
    ood_results = evaluate_chronos_on_sequences(
        pipeline, sequences_ood, val_end, L, H, batch_size=32, device=device, scenario_name="OOD_Turbulence"
    )

    final_results = {
        "Standard_Environment": standard_results,
        "OOD_Turbulence_Environment": ood_results
    }
    
    res_file = "chronos_rov_results_with_ood.json"
    with open(res_file, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nAll tests completed! Results saved to {res_file}")

if __name__ == "__main__":
    run_chronos_baseline_experiment()