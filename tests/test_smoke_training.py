"""
CPU Smoke Test verifying end-to-end scientific model training loops.
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from main_demo import EfficientScientificModel, BaselineHeavyScientificModel
from utils.training_utils import AMPTrainer, GradientAccumulator, ScientificMetricsTracker


def test_efficient_model_forward_backward_cpu():
    """Verify EfficientScientificModel runs forward, backward, and metrics on CPU."""
    # Synthetic batch: (Batch=4, Time=24, Stations=8, Features=4)
    x = torch.randn(4, 24, 8, 4)
    y = torch.randn(4, 8)

    model = EfficientScientificModel(in_features=4, seq_len=24, num_stations=8)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    preds = model(x)
    assert preds.shape == (4, 8)

    loss = loss_fn(preds, y)
    assert torch.isfinite(loss)

    loss.backward()
    opt.step()
    opt.zero_grad()


def test_amp_trainer_and_accumulator_cpu_loop():
    """Verify AMPTrainer and GradientAccumulator interface works seamlessly on CPU."""
    x = torch.randn(16, 24, 8, 4)
    y = torch.randn(16, 8)
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=4)

    model = EfficientScientificModel(in_features=4, seq_len=24, num_stations=8)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    trainer = AMPTrainer(model=model, optimizer=opt, use_amp=False)
    accumulator = GradientAccumulator(accumulation_steps=2)
    tracker = ScientificMetricsTracker()

    total_batches = len(loader)
    for idx, (bx, by) in enumerate(loader):
        should_step = accumulator.should_step(idx, total_batches)
        loss, preds = trainer.forward_backward_step(
            bx, by, loss_fn=loss_fn, accumulate_grad=(not should_step), accumulation_steps=2
        )
        tracker.update(preds, by, loss.item(), bx.size(0))

    metrics = tracker.compute()
    assert metrics["loss"] >= 0.0
    assert torch.isfinite(torch.tensor(metrics["mse"]))
