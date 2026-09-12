"""
Resilient Checkpoint Manager
============================
Provides atomic saving, automated Colab Google Drive / Kaggle synchronization,
RNG state preservation, and graceful preemption recovery.
"""

from __future__ import annotations

import os
import random
import signal
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


def setup_colab_drive_checkpointing(
    drive_folder_name: str = "SciML_Colab_Checkpoints",
) -> Path:
    """Detect if running in Google Colab, mount Google Drive, and return persistent path.

    Args:
        drive_folder_name: Folder name on Google Drive to store resilient artifacts.

    Returns:
        Path object pointing to persistent save directory.
    """
    is_colab = "google.colab" in sys.modules
    is_kaggle = os.path.exists("/kaggle/working")

    if is_colab:
        try:
            from google.colab import drive  # type: ignore

            print(
                "[Colab Setup] Mounting Google Drive to persist checkpoints across disconnections..."
            )
            drive.mount("/content/drive", force_remount=False)
            target_dir = Path("/content/drive/MyDrive") / drive_folder_name
            target_dir.mkdir(parents=True, exist_ok=True)
            print(f"[Colab Setup] Checkpoint directory established at: {target_dir}")
            return target_dir
        except Exception as e:
            print(
                f"[Colab Setup Warning] Google Drive mount failed ({e}). Falling back to local content directory."
            )
            local_dir = Path("/content/checkpoints")
            local_dir.mkdir(parents=True, exist_ok=True)
            return local_dir

    elif is_kaggle:
        print(
            "[Kaggle Setup] Kaggle kernel detected. Directing checkpoints to /kaggle/working/checkpoints..."
        )
        target_dir = Path("/kaggle/working/checkpoints")
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir

    else:
        # Standard local workstation or Docker container
        local_dir = Path("./checkpoints")
        local_dir.mkdir(parents=True, exist_ok=True)
        return local_dir


class CheckpointManager:
    """Manages atomic checkpoints, RNG states, and best-model tracking."""

    def __init__(
        self,
        checkpoint_dir: Union[str, Path] = "./checkpoints",
        max_to_keep: int = 3,
        best_metric_mode: str = "min",
        project_name: str = "sciml_experiment",
    ) -> None:
        """Initialize CheckpointManager.

        Args:
            checkpoint_dir: Directory to save checkpoint files.
            max_to_keep: Maximum number of recent checkpoints to retain on disk.
            best_metric_mode: 'min' (e.g. MSE loss) or 'max' (e.g. R2, accuracy).
            project_name: Prefix name for checkpoint filenames.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = max_to_keep
        self.best_metric_mode = best_metric_mode
        self.project_name = project_name

        self.best_metric = float("inf") if best_metric_mode == "min" else float("-inf")
        self.best_checkpoint_path: Optional[Path] = None
        self.saved_checkpoints: List[Path] = []

        # Hook preemption / termination signals to save an emergency checkpoint
        self._register_preemption_hooks()

    def _register_preemption_hooks(self) -> None:
        """Register OS signal handlers to trap preemption / disconnect signals."""
        self._emergency_payload: Optional[Dict[str, Any]] = None

        def emergency_signal_handler(signum: int, frame: Any) -> None:
            sig_name = (
                signal.Signals(signum).name
                if hasattr(signal, "Signals")
                else str(signum)
            )
            print(f"\n[EMERGENCY] Caught preemption/termination signal: {sig_name}!")
            if self._emergency_payload is not None:
                emergency_path = (
                    self.checkpoint_dir / f"{self.project_name}_emergency_preempt.pt"
                )
                print(f"[EMERGENCY] Flushed emergency checkpoint to {emergency_path}")
                save_atomic_checkpoint(self._emergency_payload, emergency_path)
            sys.exit(0)

        # Catch SIGTERM (sent by cloud container orchestrator before kill)
        # Note: on Windows, SIGTERM is available in signal module
        try:
            signal.signal(signal.SIGTERM, emergency_signal_handler)
        except (AttributeError, ValueError):
            pass

    def update_emergency_payload(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        epoch: int,
        loss: float,
        scheduler: Optional[_LRScheduler] = None,
    ) -> None:
        """Cache current state dictionary for instantaneous preemption saving."""
        self._emergency_payload = self._capture_full_state(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            metric_val=loss,
            extra_metadata={"status": "in_progress_preemption_buffer"},
        )

    def _capture_full_state(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        epoch: int,
        metric_val: float,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Capture complete model, optimizer, scheduler, and RNG states."""
        # Unpack model in case of DataParallel or DDP wrapper
        model_state = (
            model.module.state_dict()
            if hasattr(model, "module")
            else model.state_dict()
        )

        rng_states: Dict[str, Any] = {
            "python_random": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_states["torch_cuda"] = torch.cuda.get_rng_state_all()

        payload: Dict[str, Any] = {
            "epoch": epoch,
            "metric_val": metric_val,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict()
            if scheduler is not None
            else None,
            "rng_states": rng_states,
            "metadata": extra_metadata or {},
        }
        return payload

    def save(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        epoch: int,
        metric_val: float,
        scheduler: Optional[_LRScheduler] = None,
        is_best: bool = False,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save a new checkpoint atomically and maintain recent history.

        Args:
            model: PyTorch model.
            optimizer: Optimizer.
            epoch: Current completed epoch index.
            metric_val: Evaluation metric (e.g. val_loss or accuracy).
            scheduler: Optional learning rate scheduler.
            is_best: Whether this checkpoint achieved best performance.
            extra_metadata: Custom metadata dictionary.

        Returns:
            Path to the newly saved checkpoint.
        """
        payload = self._capture_full_state(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            metric_val=metric_val,
            extra_metadata=extra_metadata,
        )

        filename = f"{self.project_name}_epoch_{epoch:03d}.pt"
        checkpoint_path = self.checkpoint_dir / filename

        # 1. Atomic write to temporary file first, then atomic rename
        save_atomic_checkpoint(payload, checkpoint_path)
        self.saved_checkpoints.append(checkpoint_path)

        # 2. Update latest symlink/copy for easy resumption
        latest_path = self.checkpoint_dir / f"{self.project_name}_latest.pt"
        save_atomic_checkpoint(payload, latest_path)

        # 3. Check and save best model
        improved = False
        if self.best_metric_mode == "min" and metric_val < self.best_metric:
            improved = True
        elif self.best_metric_mode == "max" and metric_val > self.best_metric:
            improved = True

        if improved or is_best:
            self.best_metric = metric_val
            best_path = self.checkpoint_dir / f"{self.project_name}_best.pt"
            save_atomic_checkpoint(payload, best_path)
            self.best_checkpoint_path = best_path
            print(
                f"  [Checkpoint] New best model saved! (Metric: {metric_val:.6f}) -> {best_path.name}"
            )

        # 4. Prune older checkpoints to prevent disk quota exhaustion on Colab/Kaggle
        self._prune_old_checkpoints()

        return checkpoint_path

    def _prune_old_checkpoints(self) -> None:
        """Prune excess checkpoints while protecting best and latest."""
        while len(self.saved_checkpoints) > self.max_to_keep:
            oldest = self.saved_checkpoints.pop(0)
            if (
                oldest.exists()
                and "best" not in oldest.name
                and "latest" not in oldest.name
            ):
                try:
                    oldest.unlink()
                except OSError:
                    pass

    def load_latest(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[int, float, Dict[str, Any]]:
        """Load latest available checkpoint, restoring weights and RNG determinism.

        Returns:
            Tuple of (start_epoch, metric_val, metadata_dict).
        """
        latest_path = self.checkpoint_dir / f"{self.project_name}_latest.pt"
        if not latest_path.exists():
            # Check for any epoch checkpoint
            candidates = sorted(
                list(self.checkpoint_dir.glob(f"{self.project_name}_epoch_*.pt"))
            )
            if candidates:
                latest_path = candidates[-1]
            else:
                print(
                    f"[Checkpoint] No existing checkpoint found in {self.checkpoint_dir}. Starting fresh."
                )
                return 0, float("inf"), {}

        return load_resilient_checkpoint(
            checkpoint_path=latest_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )


def save_atomic_checkpoint(payload: Dict[str, Any], target_path: Path) -> None:
    """Perform atomic disk save using temporary intermediate file.

    Guarantees that Google Colab session timeouts or Kaggle disconnects
    do NOT leave a half-written, corrupted .pt file on Google Drive.

    Args:
        payload: Checkpoint dictionary containing states.
        target_path: Final destination path.
    """
    target_path = Path(target_path)
    tmp_path = target_path.with_suffix(f".tmp_{os.getpid()}")
    try:
        torch.save(payload, tmp_path)
        # os.replace is atomic across POSIX and Windows on same filesystem
        os.replace(tmp_path, target_path)
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        raise OSError(f"Atomic checkpoint save failed for {target_path}: {e}") from e


def load_resilient_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    device: Optional[torch.device] = None,
) -> Tuple[int, float, Dict[str, Any]]:
    """Load checkpoint and restore model weights, optimizer, and RNG states.

    Args:
        checkpoint_path: Path to checkpoint file.
        model: PyTorch model to receive weights.
        optimizer: Optional optimizer to restore states.
        scheduler: Optional scheduler to restore.
        device: PyTorch device mapping.

    Returns:
        Tuple of (resumed_epoch, best_metric, metadata).
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    map_location = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    # In PyTorch 2.6+, weights_only defaults to True which blocks numpy/python RNG tuple unpickling
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=map_location)

    # 1. Load Model Weights
    if hasattr(model, "module"):
        model.module.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])

    # 2. Load Optimizer States
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # 3. Load Scheduler States
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # 4. Restore RNG states to guarantee seamless determinism
    rng_states = checkpoint.get("rng_states", {})
    if "python_random" in rng_states:
        random.setstate(rng_states["python_random"])
    if "numpy" in rng_states:
        np.random.set_state(rng_states["numpy"])
    if "torch_cpu" in rng_states:
        torch.set_rng_state(rng_states["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in rng_states:
        try:
            torch.cuda.set_rng_state_all(rng_states["torch_cuda"])
        except Exception:
            pass

    epoch = checkpoint.get("epoch", 0)
    metric_val = checkpoint.get("metric_val", 0.0)
    metadata = checkpoint.get("metadata", {})

    print(
        f"[Checkpoint] Successfully resumed from {path.name} (Epoch: {epoch}, Metric: {metric_val:.6f})"
    )
    return epoch, metric_val, metadata
