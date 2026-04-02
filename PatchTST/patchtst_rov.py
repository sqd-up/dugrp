import os
import sys
import time
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Add parent directory to path to use DUGRP's datasets and evaluate modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dugrp.datasets import inject_delay, DELAY_PROFILES
from dugrp.evaluate import compute_mae, compute_rmse, compute_crps

# ---------------------------------------------------------------------------
# 1. PatchTST Model Components (with MC Dropout)
# ---------------------------------------------------------------------------
class ROVPatchTST(nn.Module):
    def __init__(self, c_in=6, context_window=64, target_window=24, patch_len=16, stride=8, d_model=64, n_heads=4, e_layers=2, dropout=0.2):
        super(ROVPatchTST, self).__init__()
        self.c_in = c_in
        self.context_window = context_window
        self.target_window = target_window
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        
        # Calculate number of patches
        self.patch_num = int((context_window - patch_len) / stride + 1)
        
        # Linear projection for patches
        self.value_embedding = nn.Linear(patch_len, d_model)
        
        # Learnable Positional Encoding
        self.position_embedding = nn.Parameter(torch.zeros(1, self.patch_num, d_model))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=256, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        
        # Explicit dropout for MC Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output prediction head (maps patches to target sequence)
        self.head = nn.Linear(self.patch_num * d_model, target_window)

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Channels]
        B, L, M = x.shape
        
        # 1. Channel Independence: treat channels as a batch dimension
        # x: [B, L, M] -> [B, M, L] -> [B * M, L, 1]
        x = x.permute(0, 2, 1).contiguous().view(B * M, L, 1)
        
        # 2. Patching
        # Unfold extracts sliding local blocks (patches)
        # patches shape: [B * M, patch_num, 1, patch_len] -> [B * M, patch_num, patch_len]
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride).squeeze(2)
        
        # 3. Patch Embedding + Positional Encoding
        # x_emb shape: [B * M, patch_num, d_model]
        x_emb = self.value_embedding(patches) + self.position_embedding
        
        # 4. Transformer Encoder
        enc_out = self.transformer_encoder(x_emb)
        
        # 5. Flatten and Output Projection
        enc_out = enc_out.reshape(B * M, -1)
        enc_out = self.dropout(enc_out)
        
        # out shape: [B * M, target_window]
        out = self.head(enc_out)
        
        # 6. Reshape back to independent channels
        # out: [B * M, target_window] -> [B, M, target_window] -> [B, target_window, M]
        out = out.view(B, M, self.target_window).permute(0, 2, 1)
        
        return out

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
def run_patchtst_experiment():
    print("=" * 60)
    print("Running Baseline: PatchTST (MC Dropout) with Generalization Test")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    L = 64
    H = 24
    patch_len = 16
    stride = 8
    
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
    model = ROVPatchTST(
        c_in=6, 
        context_window=L, 
        target_window=H, 
        patch_len=patch_len, 
        stride=stride, 
        d_model=64, 
        n_heads=4, 
        e_layers=2, 
        dropout=0.2
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print("Training started on Standard Environment...")
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

    # 1. Evaluate on Standard Environment
    standard_results = evaluate_on_sequences(
        model, sequences, val_end, L, H, batch_size, device, mc_samples, scenario_name="Standard"
    )

    # 2. Generalization Test: Simulate sudden Ocean Current
    print("\n" + "=" * 60)
    print("Applying Out-of-Distribution (OOD) Test: Simulated Ocean Current")
    print("=" * 60)
    
    sequences_ood = sequences.copy()
    drift_intensity = 1.0
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
    
    res_file = "patchtst_rov_results_with_ood.json"
    with open(res_file, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nAll tests completed! Results saved to {res_file}")

if __name__ == "__main__":
    run_patchtst_experiment()