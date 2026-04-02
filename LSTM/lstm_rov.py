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

class ROVLSTM(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, num_layers=2, pred_len=24, dropout=0.2):
        super(ROVLSTM, self).__init__()
        self.pred_len = pred_len
        self.output_dim = input_dim
        
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, pred_len * input_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        pred = self.fc(last_hidden)
        return pred.view(-1, self.pred_len, self.output_dim)

def evaluate_on_sequences(model, sequences, val_end, L, H, batch_size, device, mc_samples, scenario_name="Standard"):
    """Helper function to run evaluation on specific sequences."""
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

def run_lstm_experiment():
    print("=" * 60)
    print("Running Baseline: LSTM (MC Dropout) with Generalization Test")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
    model = ROVLSTM(input_dim=6, hidden_dim=128, num_layers=2, pred_len=H).to(device)
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
    
    # Create a deep copy of the sequences for the OOD test
    sequences_ood = sequences.copy()
    
    # Simulate a strong constant ocean current by adding a drift to X and Y velocities
    # Index 0 is X velocity, Index 1 is Y velocity. Adding 1.5 standard deviations of drift.
    drift_intensity = 1.5 
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
    
    res_file = "lstm_rov_results_with_ood.json"
    with open(res_file, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nAll tests completed! Results saved to {res_file}")

if __name__ == "__main__":
    run_lstm_experiment()