"""
Scientific ML Training Utilities
=================================
Provides production-grade training abstractions optimized for free-tier GPUs:
- AMPTrainer: Automatic Mixed Precision (FP16/BF16) with GradScaler
- GradientAccumulator: Multi-step gradient accumulation for virtual batch sizing
- Schedulers & Early Stopping tuned for physical & scientific stability
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class AMPTrainer:
    """Manages Automatic Mixed Precision (AMP) training across CUDA and CPU devices."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        device: Optional[torch.device] = None,
        use_amp: bool = True,
        amp_dtype: Optional[torch.dtype] = None,
        max_grad_norm: Optional[float] = 1.0,
    ) -> None:
        """Initialize AMP Trainer.

        Args:
            model: PyTorch neural network module.
            optimizer: Optimizer instance (e.g. AdamW, SGD).
            device: Target device. Defaults to cuda:0 if available, else cpu.
            use_amp: Whether mixed precision is enabled.
            amp_dtype: Precision format (torch.float16 or torch.bfloat16).
                       On Nvidia T4/P100, float16 with GradScaler is optimal.
                       On A100/H100, bfloat16 avoids underflow without a scaler.
            max_grad_norm: Maximum gradient norm for clipping. Crucial for stiff ODE/PDE losses.
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.is_cuda = self.device.type == "cuda"
        self.use_amp = use_amp and self.is_cuda
        self.max_grad_norm = max_grad_norm

        # Select precision type
        if amp_dtype is None:
            # T4 (Turing) and P100 (Pascal) perform best with FP16 Tensor Cores
            self.amp_dtype = torch.float16 if self.is_cuda else torch.bfloat16
        else:
            self.amp_dtype = amp_dtype

        # GradScaler is necessary for FP16 due to small dynamic range (5 exponent bits)
        # BF16 has 8 exponent bits (same as FP32), so scaling is typically unnecessary
        scaler_enabled = self.use_amp and (self.amp_dtype == torch.float16)
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    def forward_backward_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        accumulate_grad: bool = False,
        accumulation_steps: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Execute a single mixed-precision forward pass and backward propagation.

        Args:
            inputs: Batch inputs tensor.
            targets: Batch targets tensor.
            loss_fn: Loss function callable.
            accumulate_grad: If True, do not step optimizer yet.
            accumulation_steps: Divisor for gradient normalization.

        Returns:
            Tuple of (unscaled loss tensor, model predictions).
        """
        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)

        # Autocast context: cast operations like matmul & convs to FP16, reductions to FP32
        device_type = "cuda" if self.is_cuda else "cpu"
        with torch.amp.autocast(device_type=device_type, dtype=self.amp_dtype, enabled=self.use_amp):
            predictions = self.model(inputs)
            raw_loss = loss_fn(predictions, targets)
            # Normalize loss if using gradient accumulation
            scaled_loss = raw_loss / accumulation_steps

        # Backward with GradScaler to prevent gradient underflow in FP16
        if self.use_amp and self.scaler.is_enabled():
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        if not accumulate_grad:
            self.step_optimizer()

        return raw_loss.detach(), predictions.detach()

    def step_optimizer(self) -> None:
        """Step optimizer with unscaling, gradient clipping, and memory-saving zero_grad."""
        if self.use_amp and self.scaler.is_enabled():
            if self.max_grad_norm is not None:
                # Unscale gradients before gradient clipping
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

        # set_to_none=True deallocates tensor buffers rather than writing zeros
        # Yields ~5-10% memory saving and avoids a memset kernel
        self.optimizer.zero_grad(set_to_none=True)


class GradientAccumulator:
    """Manages virtual batch sizing via gradient accumulation across N micro-steps.

    Enables training on 16GB free Colab/Kaggle GPUs with an effective batch size
    equivalent to high-end enterprise clusters (e.g. 512 or 1024).
    """

    def __init__(self, accumulation_steps: int = 4) -> None:
        """Initialize accumulator.

        Args:
            accumulation_steps: Number of micro-batches to accumulate before stepping.
        """
        if accumulation_steps < 1:
            raise ValueError(f"accumulation_steps must be >= 1, got {accumulation_steps}")
        self.accumulation_steps = accumulation_steps
        self.step_count = 0

    def should_step(self, batch_idx: int, total_batches: int) -> bool:
        """Check whether current batch index triggers an optimizer step.

        Args:
            batch_idx: Current 0-indexed batch counter.
            total_batches: Total batches in epoch (steps on final mini-batch).

        Returns:
            Boolean indicating if optimizer should step.
        """
        is_accum_boundary = ((batch_idx + 1) % self.accumulation_steps == 0)
        is_final_batch = ((batch_idx + 1) == total_batches)
        return is_accum_boundary or is_final_batch


def create_scientific_lr_scheduler(
    optimizer: Optimizer,
    total_steps: int,
    warmup_steps: int = 100,
    min_lr_ratio: float = 1e-3,
) -> LambdaLR:
    """Create a Cosine Annealing scheduler with linear warmup.

    Crucial for scientific models: Warmup prevents chaotic initial gradient spikes
    from exploding stiff dynamical systems or physical constraints.

    Args:
        optimizer: Wrapped PyTorch optimizer.
        total_steps: Total training steps across all epochs.
        warmup_steps: Number of initial linear warmup steps.
        min_lr_ratio: Lowest learning rate relative to base lr.

    Returns:
        Configured LambdaLR scheduler.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


class EarlyStopping:
    """Monitors validation metric to halt training and preserve best model weights."""

    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 1e-4,
        mode: str = "min",
        verbose: bool = True,
    ) -> None:
        """Initialize EarlyStopping.

        Args:
            patience: Number of epochs without improvement before stopping.
            min_delta: Minimum significant change threshold.
            mode: 'min' for loss, 'max' for R² or accuracy.
            verbose: If True, prints messages on improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.is_best = False

    def step(self, current_metric: float) -> bool:
        """Update tracker with latest metric score.

        Args:
            current_metric: Current epoch validation metric.

        Returns:
            True if training should stop, False otherwise.
        """
        score = -current_metric if self.mode == "min" else current_metric

        if self.best_score is None:
            self.best_score = score
            self.is_best = True
            return False

        if score < self.best_score + self.min_delta:
            self.counter += 1
            self.is_best = False
            if self.verbose:
                print(f"  [EarlyStopping] No improvement ({self.counter}/{self.patience} patience).")
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_score = score
            self.counter = 0
            self.is_best = True
            if self.verbose:
                print(f"  [EarlyStopping] Metric improved! Best score: {abs(self.best_score):.6f}")

        return False


class ScientificMetricsTracker:
    """Calculates scientific regression and accuracy metrics (MSE, MAE, R2)."""

    def __init__(self) -> None:
        """Initialize metric accumulators."""
        self.reset()

    def reset(self) -> None:
        """Reset running totals."""
        self.total_samples = 0
        self.total_loss = 0.0
        self.all_preds: List[torch.Tensor] = []
        self.all_targets: List[torch.Tensor] = []

    def update(self, preds: torch.Tensor, targets: torch.Tensor, loss: float, batch_size: int) -> None:
        """Accumulate predictions and targets for global metric calculation.

        Args:
            preds: Predictions tensor.
            targets: Targets tensor.
            loss: Scaled batch loss value.
            batch_size: Number of samples in batch.
        """
        self.total_samples += batch_size
        self.total_loss += loss * batch_size
        self.all_preds.append(preds.detach().cpu())
        self.all_targets.append(targets.detach().cpu())

    def compute(self) -> Dict[str, float]:
        """Compute final epoch metrics: Average Loss, MSE, MAE, and R-squared.

        Returns:
            Dictionary of computed scientific evaluation metrics.
        """
        if self.total_samples == 0:
            return {"loss": 0.0, "mse": 0.0, "mae": 0.0, "r2": 0.0}

        mean_loss = self.total_loss / self.total_samples
        y_pred = torch.cat(self.all_preds, dim=0).float()
        y_true = torch.cat(self.all_targets, dim=0).float()

        mse = torch.mean((y_pred - y_true) ** 2).item()
        mae = torch.mean(torch.abs(y_pred - y_true)).item()

        # Coefficient of Determination (R²)
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2).item()
        ss_res = torch.sum((y_true - y_pred) ** 2).item()
        r2 = 1.0 - (ss_res / (ss_tot + 1e-8))

        return {
            "loss": mean_loss,
            "mse": mse,
            "mae": mae,
            "r2": r2,
        }
