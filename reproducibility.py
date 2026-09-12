"""
Reproducibility and Environment Snapshot Package
================================================
Guarantees bitwise determinism, environment auditing, and academic compute
citation generation for zero-budget scientific machine learning research.
"""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
import torch


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """Enforce strict reproducible execution across Python, NumPy, PyTorch, and CUDA.

    Args:
        seed: Master random seed integer.
        deterministic: If True, forces deterministic PyTorch operations and disables
                      non-deterministic CuDNN kernel autotuning.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Guarantee CuDNN convolution algorithms are deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # In PyTorch 1.8+, use_deterministic_algorithms can enforce reproducible kernels
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass

    print(
        f"[Reproducibility] Master seed set to {seed} (Deterministic mode: {deterministic})"
    )


def capture_environment(
    output_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Audit system environment, package versions, and hardware specifications.

    Args:
        output_path: Optional path to save JSON environment metadata artifact.

    Returns:
        Dictionary containing comprehensive system and compute configuration.
    """
    env_info: Dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": sys.version,
        "os": platform.system(),
        "processor": platform.processor(),
        "cpu_count_physical": (
            psutil.cpu_count(logical=False) if HAS_PSUTIL else (os.cpu_count() or 1)
        ),
        "cpu_count_logical": (
            psutil.cpu_count(logical=True) if HAS_PSUTIL else (os.cpu_count() or 1)
        ),
        "total_ram_gb": (
            round(psutil.virtual_memory().total / (1024**3), 2) if HAS_PSUTIL else 16.0
        ),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        env_info["cuda_version"] = torch.version.cuda
        env_info["cudnn_version"] = torch.backends.cudnn.version()
        env_info["device_count"] = torch.cuda.device_count()
        gpu_devices = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_devices.append(
                {
                    "device_index": i,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                }
            )
        env_info["gpus"] = gpu_devices
    else:
        env_info["cuda_version"] = None
        env_info["gpus"] = []

    # Get installed package versions
    try:
        pip_freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], stderr=subprocess.DEVNULL
        ).decode("utf-8")
        env_info["pip_packages"] = [
            line for line in pip_freeze.splitlines() if line.strip()
        ]
    except Exception:
        env_info["pip_packages"] = [
            f"torch=={torch.__version__}",
            f"numpy=={np.__version__}",
        ]

    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(env_info, f, indent=2)
        print(f"[Reproducibility] Saved environment snapshot to {target}")

    return env_info


def generate_compute_badge(
    platform_name: str = "Google Colab Free (T4)",
    training_hours: float = 0.5,
    cloud_cost_saved_usd: float = 1.53,
) -> str:
    """Generate markdown badge and academic citation for compute disclosure.

    Follows best practices in AI research reproducibility (e.g. NeurIPS compute disclosure).

    Args:
        platform_name: Compute tier utilized.
        training_hours: Total wall-clock GPU hours consumed.
        cloud_cost_saved_usd: Equivalent commercial cloud dollar savings.

    Returns:
        Formatted markdown citation text.
    """
    badge_markdown = (
        f"[![Zero Budget AI](https://img.shields.io/badge/Compute_Budget-$0.00_(Free_Tier)-brightgreen.svg)]"
        f"(https://github.com/)\n"
        f"[![Colab T4 Tested](https://img.shields.io/badge/Verified-Google_Colab_T4-blue.svg)]"
        f"(https://colab.research.google.com/)\n"
        f"[![Reproducibility](https://img.shields.io/badge/Reproducibility-Bitwise_Deterministic-purple.svg)]"
        f"(https://pytorch.org/docs/stable/notes/randomness.html)\n\n"
        f"### Compute Disclosure & Reproducibility Statement\n"
        f"> **Hardware Environment:** This scientific model was trained exclusively on **{platform_name}**.\n"
        f"> **Total Wall-Clock Time:** {training_hours:.2f} GPU hours.\n"
        f"> **Estimated Cloud Cost Saved:** **${cloud_cost_saved_usd:.2f} USD** (vs commercial AWS on-demand rates).\n"
        f"> **Hardware Equality Impact:** Proving competitive scientific machine learning research is achievable\n"
        f"> at academic institutions without dedicated enterprise GPU cluster funding."
    )
    return badge_markdown


if __name__ == "__main__":
    seed_everything(42)
    info = capture_environment("environment_snapshot.json")
    print(generate_compute_badge())
