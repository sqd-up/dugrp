import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2Config, GPT2Model
import urllib3

# Disable annoying SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

# Add parent directory to path to use DUGRP's datasets and evaluate modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dugrp.datasets import inject_delay, DELAY_PROFILES
from dugrp.evaluate import compute_mae, compute_rmse, compute_crps

# ---------------------------------------------------------------------------
# 1. TimeRAF Architecture: Local TimeGPT Surrogate + Retrieval-Augmented Fusion
# ---------------------------------------------------------------------------
class ROVTimeRAF(nn.Module):
    def __init__(self, input_dim=6, seq_len=64, pred_len=24, d_model=64, top_k=5, dropout=0.2):
        super(ROVTimeRAF, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.top_k = top_k
        self.d_model = d_model

        # 1. Local "TimeGPT" Surrogate (Causal Transformer Backbone)
        # We instantiate a local GPT2 architecture avoiding online weight downloads
        config = GPT2Config(
            vocab_size=1, # Bypass standard token embedding
            n_positions=seq_len + 10,
            n_embd=d_model,
            n_layer=3,    # Lightweight for fast local training
            n_head=4,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout
        )
        self.backbone = GPT2Model(config)
        self.in_proj = nn.Linear(input_dim, d_model)
        self.out_proj = nn.Linear(seq_len * d_model, pred_len * input_dim)

        # 2. Retrieval Feature Extractor (RAF Module)
        self.retrieval_encoder = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.Flatten(),
            nn.Linear(seq_len * d_model, d_model)
        )

        # 3. Dynamic Fusion Parameter
        # Learns to balance Backbone prediction vs Retrieved prediction
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        
        self.dropout = nn.Dropout(dropout)
        
        # Memory Bank variables
        self.memory_keys = None
        self.memory_values = None

    def load_memory(self, keys_tensor, values_tensor):
        """Build the historical knowledge base after training."""
        print(f"Building RAF Memory Bank with {keys_tensor.size(0)} sequences...")
        self.eval()
        with torch.no_grad():
            # Extract query features for the entire memory bank
            keys_emb = self.retrieval_encoder(keys_tensor.to(self.fusion_weight.device))
            self.memory_keys = F.normalize(keys_emb, p=2, dim=1)
            self.memory_values = values_tensor.to(self.fusion_weight.device)
        self.train() # Switch back to train for MC Dropout

    def forward(self, x):
        B = x.size(0)

        # === Step 1: Base "TimeGPT" Prediction ===
        x_emb = self.in_proj(x)
        # Pass continuous embeddings to GPT
        gpt_out = self.backbone(inputs_embeds=x_emb).last_hidden_state
        gpt_out = gpt_out.reshape(B, -1)
        gpt_out = self.dropout(gpt_out)
        
        base_pred = self.out_proj(gpt_out)
        base_pred = base_pred.view(B, self.pred_len, self.input_dim)

        # === Step 2: RAF (Retrieval Augmented Forecasting) ===
        if self.memory_keys is not None and self.top_k > 0:
            # Encode current query context
            q = self.retrieval_encoder(x)
            q = F.normalize(q, p=2, dim=1)

            # Cosine similarity with memory bank
            sim = torch.matmul(q, self.memory_keys.T)  # Shape: (B, M)

            # Retrieve Top-K
            topk_sim, topk_idx = torch.topk(sim, self.top_k, dim=1)
            
            # Weighted average of retrieved future values
            weights = F.softmax(topk_sim, dim=1) # Shape: (B, K)
            
            retrieved_preds = []
            for i in range(B):
                idx = topk_idx[i]
                vals = self.memory_values[idx] # (K, pred_len, dim)
                w = weights[i].view(-1, 1, 1)
                weighted_val = (vals * w).sum(dim=0)
                retrieved_preds.append(weighted_val)
                
            retrieved_pred = torch.stack(retrieved_preds)

            # === Step 3: Dynamic Fusion ===
            alpha = torch.sigmoid(self.fusion_weight)
            final_pred = alpha * base_pred + (1 - alpha) * retrieved_pred
            return final_pred
        else:
            return base_pred

# ---------------------------------------------------------------------------
# 2. Evaluation Helper
# ---------------------------------------------------------------------------
def evaluate_on_sequences(model, sequences, val_end, L, H, batch_size, device, mc_samples, scenario_name="Standard"):
    print(f"\n--- Starting Evaluation: {scenario_name} Environment ---")
    results = {}
    model.train()  # Keep dropout active for MC Dropout

    for profile_name in ["D1", "D2", "D3"]:
        profile = DELAY_PROFILES[profile_name]
        ctx_list, fut_list = [], []
        
        for seq in sequences[val_end:]:
            delayed, _ = inject_delay(seq, profile, dt_ms=100.0)
            for t in range(L, len(seq) - H + 1, 5):
                ctx_list.append(delayed[t - L : t])
                fut_list.append(seq[t : t + H])
                
        X_test = torch.tensor(np.array(ctx_list), dtype=torch.float32).to(device)
        Y_test = np.array(fut_list).astype(np.float32)
        
        test_dataset = TensorDataset(X_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        mae_list, rmse_list, crps_list = [], [], []
        
        idx = 0
        with torch.no_grad():
            for (bx,) in test_loader:
                mc_preds = []
                for _ in range(mc_samples):
                    mc_preds.append(model(bx).cpu().numpy())
                
                mc_preds = np.stack(mc_preds)
                mean_pred = mc_preds.mean(axis=0)
                q_low = np.percentile(mc_preds, 10, axis=0)
                q_high = np.percentile(mc_preds, 90, axis=0)
                
                batch_y_true = Y_test[idx : idx + bx.shape[0]]
                idx += bx.shape[0]
                
                for i in range(bx.shape[0]):
                    mae_list.append(compute_mae(mean_pred[i], batch_y_true[i]))
                    rmse_list.append(compute_rmse(mean_pred[i], batch_y_true[i]))
                    crps_list.append(compute_crps(q_low[i], q_high[i], batch_y_true[i]))
                    
        res = {
            "MAE": float(np.mean(mae_list)),
            "RMSE": float(np.mean(rmse_list)),
            "CRPS": float(np.mean(crps_list))
        }
        results[profile_name] = res
        print(f"[{scenario_name} - {profile_name}] MAE: {res['MAE']:.4f} | RMSE: {res['RMSE']:.4f} | CRPS: {res['CRPS']:.4f}")
    
    return results

# ---------------------------------------------------------------------------
# 3. Main Experiment Protocol
# ---------------------------------------------------------------------------
def run_timeraf_experiment():
    print("=" * 60)
    print("Running Baseline: TimeRAF (TimeGPT Surrogate + Retrieval Fusion)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    L = 64
    H = 24
    batch_size = 64
    epochs = 50
    lr = 1e-3
    data_path = "../data/rov_data.npy"
    mc_samples = 20

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Data loading and preprocessing
    sequences_raw = np.load(data_path)
    sequences = np.diff(sequences_raw, axis=1)
    
    N_total = len(sequences)
    train_end = int(N_total * 0.7)
    val_end   = int(N_total * 0.8)

    train_data_raw = sequences[:train_end]
    mean = train_data_raw.mean(axis=(0, 1), keepdims=True)
    std  = train_data_raw.std(axis=(0, 1), keepdims=True) + 1e-8
    
    sequences = (sequences - mean) / std

    # Build Training Set
    X_train_list, Y_train_list = [], []
    for seq in sequences[:train_end]:
        for t in range(L, len(seq) - H + 1, 5):
            X_train_list.append(seq[t - L : t])
            Y_train_list.append(seq[t : t + H])
            
    X_train = torch.tensor(np.array(X_train_list), dtype=torch.float32)
    Y_train = torch.tensor(np.array(Y_train_list), dtype=torch.float32)
    
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Model Setup
    model = ROVTimeRAF(
        input_dim=6, 
        seq_len=L, 
        pred_len=H, 
        d_model=64, 
        top_k=5, 
        dropout=0.2
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print("Training TimeRAF backbone started...")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.4f}")

    # =======================================================
    # CRITICAL: Build RAF Memory Bank before Evaluation
    # =======================================================
    model.load_memory(X_train, Y_train)

    # 1. Evaluate on Standard Environment
    standard_results = evaluate_on_sequences(
        model, sequences, val_end, L, H, batch_size, device, mc_samples, scenario_name="Standard"
    )

    # 2. Generalization Test: Simulate sudden Ocean Current
    print("\n" + "=" * 60)
    print("Applying Out-of-Distribution (OOD) Test: Simulated Ocean Current")
    print("=" * 60)
    
    sequences_ood = sequences.copy()
    drift_intensity = 0.8
    sequences_ood[val_end:, :, 0] += drift_intensity  
    sequences_ood[val_end:, :, 1] += drift_intensity  
    
    ood_results = evaluate_on_sequences(
        model, sequences_ood, val_end, L, H, batch_size, device, mc_samples, scenario_name="OOD_Ocean_Current"
    )

    # Save Results
    final_results = {
        "Standard_Environment": standard_results,
        "OOD_Ocean_Current_Environment": ood_results
    }
    
    res_file = "time_raf_rov_results_with_ood.json"
    with open(res_file, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nAll tests completed! Results saved to {res_file}")

if __name__ == "__main__":
    run_timeraf_experiment()