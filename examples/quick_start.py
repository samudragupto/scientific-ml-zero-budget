"""
Quick Start Guide: End-to-End Scientific Model Training with AMP and Checkpointing
==================================================================================
Demonstrates how to integrate Automatic Mixed Precision (AMP), gradient accumulation,
and resilient atomic checkpointing in 60 lines of clean Python.
"""

import sys
from pathlib import Path

# Add project root to sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import torch
import torch.nn as nn
from utils import (
    generate_synthetic_climate_data,
    MemmapScientificDataset,
    get_optimized_dataloader,
    AMPTrainer,
    GradientAccumulator,
    CheckpointManager,
)
from main_demo import EfficientScientificModel

def quick_start():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Scientific ML Zero Budget Quickstart on {device} ---")

    # Data setup
    feat, targ = generate_synthetic_climate_data(output_dir="./quickstart_data", num_samples=1000)
    dataset = MemmapScientificDataset(feat, targ, shape_x=(1000, 24, 8, 4), shape_y=(1000, 8))
    loader = get_optimized_dataloader(dataset, batch_size=32)

    # Model & optimizer
    model = EfficientScientificModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Utilities: AMP Trainer, 4-step Gradient Accumulation, and Atomic Checkpoints
    trainer = AMPTrainer(model=model, optimizer=optimizer, device=device, use_amp=True)
    accumulator = GradientAccumulator(accumulation_steps=4)  # 32 * 4 = virtual batch 128
    checkpointer = CheckpointManager(checkpoint_dir="./quickstart_checkpoints", project_name="sciml_quick")

    print("Training 1 epoch with AMP and virtual batch size 128...")
    model.train()
    total_loss = 0.0
    total_batches = len(loader)

    for i, (bx, by) in enumerate(loader):
        should_step = accumulator.should_step(i, total_batches)
        loss, preds = trainer.forward_backward_step(
            bx, by, loss_fn=loss_fn, accumulate_grad=(not should_step), accumulation_steps=4
        )
        total_loss += loss.item() * bx.size(0)

    mean_loss = total_loss / len(dataset)
    print(f"Epoch finished. Loss: {mean_loss:.4f}")

    # Save atomic checkpoint
    saved_path = checkpointer.save(model, optimizer, epoch=1, metric_val=mean_loss, is_best=True)
    print(f"Checkpoint safely written to: {saved_path}")

if __name__ == "__main__":
    quick_start()
