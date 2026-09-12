"""
Minimal Working Example: Scientific ML on Zero Budget
=====================================================
A concise, 40-line minimal script demonstrating memory-mapped data streaming
and training with an efficient scientific architecture.
"""

import sys
from pathlib import Path

# Add project root to sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import torch
import torch.nn as nn
from utils.data_streaming import generate_synthetic_climate_data, MemmapScientificDataset, get_optimized_dataloader
from main_demo import EfficientScientificModel

def run_minimal():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Minimal Scientific ML Example on: {device}")

    # 1. Generate 500 samples of spatio-temporal atmospheric data on disk
    feat_p, targ_p = generate_synthetic_climate_data(output_dir="./data_minimal", num_samples=500)
    dataset = MemmapScientificDataset(feat_p, targ_p, shape_x=(500, 24, 8, 4), shape_y=(500, 8))
    loader = get_optimized_dataloader(dataset, batch_size=32)

    # 2. Instantiate efficient scientific model (DS-Conv + SE attention)
    model = EfficientScientificModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # 3. Train for 1 epoch
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)

    print(f"Epoch Complete! Mean MSE Loss: {total_loss / len(dataset):.4f}")

if __name__ == "__main__":
    run_minimal()
