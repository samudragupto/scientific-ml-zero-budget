"""
One-Click Google Colab / Kaggle Setup Script
============================================
Detects execution environment, verifies GPU acceleration, mounts Google Drive,
configures persistent directories, and validates CUDA tensor core availability.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def is_running_in_colab() -> bool:
    """Return True if executing within Google Colab environment."""
    return "google.colab" in sys.modules or "COLAB_GPU" in os.environ


def is_running_in_kaggle() -> bool:
    """Return True if executing within a Kaggle Notebook container."""
    return os.path.exists("/kaggle/working")


def setup_environment(
    seed: int = 42,
    drive_folder: str = "SciML_Colab_Checkpoints",
) -> Path:
    """Perform complete automated environment initialization for Colab/Kaggle.

    Args:
        seed: Master random seed.
        drive_folder: Google Drive folder name for persistent saves.

    Returns:
        Path to checkpoint directory.
    """
    print("=" * 75)
    print("      SCIENTIFIC ML ZERO BUDGET: AUTOMATED ENVIRONMENT SETUP      ")
    print("=" * 75)

    # 1. Environment Detection
    in_colab = is_running_in_colab()
    in_kaggle = is_running_in_kaggle()

    if in_colab:
        print("[Environment] Detected Google Colab Runtime.")
    elif in_kaggle:
        print("[Environment] Detected Kaggle Notebook Runtime.")
    else:
        print("[Environment] Detected Local Workstation / Docker Container.")

    # 2. PyTorch & CUDA Verification
    try:
        import torch

        print(f"[PyTorch] Installed Version: {torch.__version__}")
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            compute_cap = torch.cuda.get_device_capability(0)
            print(f"[CUDA Status] SUCCESS! Detected GPU: {device_name}")
            print(f"[CUDA Status] Total Available VRAM: {vram_gb:.2f} GB")
            print(f"[CUDA Status] Compute Architecture: SM {compute_cap[0]}.{compute_cap[1]}")

            if compute_cap[0] >= 7:
                print("  -> Tensor Cores active! Mixed Precision (AMP FP16) fully accelerated.")
            if compute_cap[0] >= 8:
                print("  -> Ampere+ architecture detected: Native BF16 support available.")
        else:
            print("[CUDA Status] WARNING: No GPU detected! Running in CPU emulation mode.")
            if in_colab:
                print("  -> Colab Instruction: Go to 'Runtime' -> 'Change runtime type' -> Select 'T4 GPU'.")
            elif in_kaggle:
                print("  -> Kaggle Instruction: Toggle 'Accelerator' -> 'GPU P100' or 'GPU T4 x2' in settings.")
    except ImportError:
        print("[ERROR] PyTorch is not installed. Please install torch>=2.0.0.")

    # 3. Mount Google Drive if in Colab
    checkpoint_dir: Path
    if in_colab:
        try:
            from google.colab import drive  # type: ignore

            print("[Storage] Mounting Google Drive to survive 12-hour session disconnects...")
            drive.mount("/content/drive", force_remount=False)
            checkpoint_dir = Path("/content/drive/MyDrive") / drive_folder
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            print(f"[Storage] Persistent Google Drive folder: {checkpoint_dir}")
        except Exception as e:
            print(f"[Storage Warning] Could not mount Drive ({e}). Using local /content/checkpoints.")
            checkpoint_dir = Path("/content/checkpoints")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
    elif in_kaggle:
        checkpoint_dir = Path("/kaggle/working/checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Storage] Kaggle persistent working directory: {checkpoint_dir}")
    else:
        checkpoint_dir = Path("./checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Storage] Local checkpoint directory: {checkpoint_dir.resolve()}")

    # 4. Set Master Seed
    from reproducibility import seed_everything

    seed_everything(seed=seed, deterministic=True)

    print("=" * 75)
    print("  ENVIRONMENT SETUP COMPLETE: Ready for Zero-Budget Scientific ML!  ")
    print("=" * 75)
    return checkpoint_dir


if __name__ == "__main__":
    setup_environment()
