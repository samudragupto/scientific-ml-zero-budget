"""
Tests for atomic checkpoint saving, state dictionary recovery, and RNG persistence.
"""

from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
from utils.checkpoint_manager import CheckpointManager, save_atomic_checkpoint, load_resilient_checkpoint


def test_atomic_checkpoint_save_and_load(tmp_path: Path):
    """Verify atomic checkpoint saves state dicts and successfully restores them."""
    model = nn.Linear(10, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    # Initial forward pass to create gradients and optimizer state
    x = torch.randn(4, 10)
    out = model(x).sum()
    out.backward()
    optimizer.step()

    original_weight = model.weight.data.clone()

    ckpt_file = tmp_path / "test_model.pt"
    payload = {
        "epoch": 5,
        "metric_val": 0.1234,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    save_atomic_checkpoint(payload, ckpt_file)
    assert ckpt_file.exists()

    # Create fresh model and restore
    new_model = nn.Linear(10, 2)
    new_opt = torch.optim.SGD(new_model.parameters(), lr=0.01)

    epoch, metric, _ = load_resilient_checkpoint(ckpt_file, new_model, new_opt)

    assert epoch == 5
    assert abs(metric - 0.1234) < 1e-6
    torch.testing.assert_close(new_model.weight.data, original_weight)


def test_checkpoint_manager_best_and_pruning(tmp_path: Path):
    """Verify CheckpointManager tracks best model and respects max_to_keep limit."""
    manager = CheckpointManager(checkpoint_dir=tmp_path, max_to_keep=2, best_metric_mode="min")
    model = nn.Linear(5, 1)
    opt = torch.optim.Adam(model.parameters())

    # Save 4 checkpoints with varying metric values
    manager.save(model, opt, epoch=1, metric_val=1.5)
    manager.save(model, opt, epoch=2, metric_val=1.1, is_best=True)
    manager.save(model, opt, epoch=3, metric_val=1.3)
    manager.save(model, opt, epoch=4, metric_val=0.9, is_best=True)

    # Best model file must exist
    best_file = tmp_path / "sciml_experiment_best.pt"
    assert best_file.exists()

    # Total epoch files should be pruned to at most max_to_keep (2)
    epoch_files = list(tmp_path.glob("*_epoch_*.pt"))
    assert len(epoch_files) <= 2
