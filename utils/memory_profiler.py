"""
Memory Profiler and Efficiency Metrics Module
=============================================
Provides precise GPU VRAM and system CPU RAM tracking, context managers,
decorators, and OOM risk heuristics tailored for resource-constrained
environments like Google Colab and Kaggle.
"""

from __future__ import annotations

import functools
import gc
import os
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, Optional, Tuple

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
import torch


class MemoryTracker:
    """Tracks GPU VRAM and CPU RAM consumption metrics during execution."""

    def __init__(self, device: Optional[torch.device] = None) -> None:
        """Initialize tracker with target PyTorch device.

        Args:
            device: Optional torch.device. If None, auto-selects cuda:0 if available, else cpu.
        """
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.is_cuda = self.device.type == "cuda"
        self.process = psutil.Process(os.getpid()) if HAS_PSUTIL else None

    def reset(self) -> None:
        """Reset peak memory tracking statistics."""
        gc.collect()
        if self.is_cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)

    def get_cpu_ram_mb(self) -> float:
        """Return current process resident set size (RSS) RAM in megabytes."""
        if HAS_PSUTIL and self.process is not None:
            return self.process.memory_info().rss / (1024.0 * 1024.0)
        return 0.0

    def get_gpu_ram_mb(self) -> Tuple[float, float, float]:
        """Return (allocated_mb, reserved_mb, peak_allocated_mb) for target GPU device.

        Returns:
            Tuple of (allocated_mb, reserved_mb, peak_allocated_mb).
            All zeros if running on CPU.
        """
        if not self.is_cuda:
            return 0.0, 0.0, 0.0

        allocated = torch.cuda.memory_allocated(self.device) / (1024.0 * 1024.0)
        reserved = torch.cuda.memory_reserved(self.device) / (1024.0 * 1024.0)
        peak = torch.cuda.max_memory_allocated(self.device) / (1024.0 * 1024.0)
        return allocated, reserved, peak

    def get_hardware_info(self) -> Dict[str, Any]:
        """Return structured summary of available hardware resources."""
        if HAS_PSUTIL:
            sys_ram = psutil.virtual_memory()
            ram_tot = sys_ram.total / (1024.0**3)
            ram_avail = sys_ram.available / (1024.0**3)
            cpu_cnt = psutil.cpu_count(logical=True)
        else:
            ram_tot = 16.0
            ram_avail = 12.0
            cpu_cnt = os.cpu_count() or 2

        info: Dict[str, Any] = {
            "device": str(self.device),
            "is_cuda": self.is_cuda,
            "cpu_count_logical": cpu_cnt,
            "cpu_ram_total_gb": ram_tot,
            "cpu_ram_available_gb": ram_avail,
        }

        if self.is_cuda:
            props = torch.cuda.get_device_properties(self.device)
            info.update(
                {
                    "gpu_name": props.name,
                    "gpu_total_vram_gb": props.total_memory / (1024.0**3),
                    "gpu_compute_capability": f"{props.major}.{props.minor}",
                    "gpu_multiprocessor_count": props.multi_processor_count,
                }
            )
        else:
            info.update(
                {
                    "gpu_name": "None (Running on CPU)",
                    "gpu_total_vram_gb": 0.0,
                    "gpu_compute_capability": "N/A",
                }
            )

        return info


@contextmanager
def profile_memory(
    label: str = "Execution Block",
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Generator[Dict[str, float], None, None]:
    """Context manager for profiling memory usage and execution latency.

    Args:
        label: Descriptive label for the code block being profiled.
        device: PyTorch device (CPU or CUDA).
        verbose: Whether to print benchmark results upon exit.

    Yields:
        Dictionary reference populated with metrics upon completion:
            - time_seconds
            - peak_gpu_mb
            - delta_gpu_alloc_mb
            - delta_cpu_ram_mb
            - peak_cpu_ram_mb
    """
    tracker = MemoryTracker(device=device)
    tracker.reset()

    start_cpu_ram = tracker.get_cpu_ram_mb()
    start_gpu_alloc, _, _ = tracker.get_gpu_ram_mb()

    if tracker.is_cuda:
        torch.cuda.synchronize(tracker.device)
    start_time = time.perf_counter()

    metrics: Dict[str, float] = {
        "time_seconds": 0.0,
        "peak_gpu_mb": 0.0,
        "delta_gpu_alloc_mb": 0.0,
        "delta_cpu_ram_mb": 0.0,
        "peak_cpu_ram_mb": start_cpu_ram,
    }

    try:
        yield metrics
    finally:
        if tracker.is_cuda:
            torch.cuda.synchronize(tracker.device)
        elapsed = time.perf_counter() - start_time

        end_cpu_ram = tracker.get_cpu_ram_mb()
        end_gpu_alloc, _, peak_gpu_alloc = tracker.get_gpu_ram_mb()

        metrics["time_seconds"] = elapsed
        metrics["peak_gpu_mb"] = peak_gpu_alloc
        metrics["delta_gpu_alloc_mb"] = max(0.0, end_gpu_alloc - start_gpu_alloc)
        metrics["delta_cpu_ram_mb"] = max(0.0, end_cpu_ram - start_cpu_ram)
        metrics["peak_cpu_ram_mb"] = end_cpu_ram

        if verbose:
            print(f"\n[{label}] Profile Report:")
            print(f"  Execution Time    : {elapsed:.4f} seconds")
            if tracker.is_cuda:
                print(f"  Peak GPU VRAM     : {peak_gpu_alloc:.2f} MB")
                print(f"  Delta GPU Alloc   : {metrics['delta_gpu_alloc_mb']:.2f} MB")
            print(f"  Current CPU RAM   : {end_cpu_ram:.2f} MB (Delta: +{metrics['delta_cpu_ram_mb']:.2f} MB)")


def benchmark_efficiency(
    num_samples: Optional[int] = None,
    hourly_cloud_cost_usd: float = 3.06,  # AWS p3.2xlarge V100 baseline reference
) -> Callable:
    """Decorator to measure execution speed, memory footprint, and compute savings.

    Args:
        num_samples: Total dataset samples processed by decorated function.
        hourly_cloud_cost_usd: Estimated hourly cloud commercial GPU cost (default: $3.06/hr).

    Returns:
        Decorated callable returning (result, benchmark_metrics_dict).
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
            tracker = MemoryTracker()
            tracker.reset()

            start_cpu = tracker.get_cpu_ram_mb()
            if tracker.is_cuda:
                torch.cuda.synchronize(tracker.device)
            t0 = time.perf_counter()

            result = func(*args, **kwargs)

            if tracker.is_cuda:
                torch.cuda.synchronize(tracker.device)
            t1 = time.perf_counter()
            elapsed_sec = t1 - t0

            end_cpu = tracker.get_cpu_ram_mb()
            _, _, peak_gpu = tracker.get_gpu_ram_mb()

            throughput = (num_samples / elapsed_sec) if num_samples and elapsed_sec > 0 else 0.0
            hours_consumed = elapsed_sec / 3600.0
            cost_saved_usd = hours_consumed * hourly_cloud_cost_usd

            metrics: Dict[str, Any] = {
                "elapsed_seconds": elapsed_sec,
                "peak_gpu_vram_mb": peak_gpu,
                "cpu_ram_mb": end_cpu,
                "throughput_samples_per_sec": throughput,
                "estimated_cloud_cost_saved_usd": cost_saved_usd,
            }
            return result, metrics

        return wrapper

    return decorator


def estimate_oom_risk(
    model: torch.nn.Module,
    sample_input_shape: Tuple[int, ...],
    batch_size: int,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
    safety_margin: float = 1.3,
) -> Dict[str, Any]:
    """Heuristically estimate whether a batch size risks Out-Of-Memory (OOM).

    Calculates:
        1. Model parameters memory footprint
        2. Gradient memory footprint (1x param size)
        3. Optimizer state footprint (AdamW = 2x param size)
        4. Forward activation memory estimation based on input dimensions

    Args:
        model: PyTorch model module.
        sample_input_shape: Tensor shape for a single sample (excluding batch dim).
        batch_size: Candidate training batch size.
        dtype: Training data type (float32, float16, bfloat16).
        device: PyTorch device.
        safety_margin: Multiplier for PyTorch caching allocator overhead.

    Returns:
        Dictionary with memory estimates and 'safe' (bool) flag.
    """
    tracker = MemoryTracker(device=device)
    bytes_per_elem = 2 if dtype in (torch.float16, torch.bfloat16) else 4

    # 1. Parameter count & weights memory
    num_params = sum(p.numel() for p in model.parameters())
    param_mb = (num_params * bytes_per_elem) / (1024.0 * 1024.0)

    # 2. Gradient memory (typically matches parameter size)
    grad_mb = param_mb

    # 3. Optimizer memory (e.g. AdamW keeps fp32 m & v moments: 8 bytes per param)
    opt_mb = (num_params * 8) / (1024.0 * 1024.0)

    # 4. Rough activation memory: intermediate feature activations scale with batch size
    elements_per_sample = 1
    for dim in sample_input_shape:
        elements_per_sample *= dim
    # In deep neural nets, intermediate activations typically span 10x-50x the input tensor size
    activation_multiplier = 30.0
    activation_mb = (
        batch_size * elements_per_sample * bytes_per_elem * activation_multiplier
    ) / (1024.0 * 1024.0)

    total_estimated_mb = (param_mb + grad_mb + opt_mb + activation_mb) * safety_margin

    if tracker.is_cuda:
        total_available_mb = (
            torch.cuda.get_device_properties(tracker.device).total_memory / (1024.0 * 1024.0)
        )
    else:
        total_available_mb = (psutil.virtual_memory().available / (1024.0 * 1024.0)) if HAS_PSUTIL else 8192.0

    is_safe = total_estimated_mb < (total_available_mb * 0.90)  # keep 10% safety buffer

    return {
        "is_safe": is_safe,
        "estimated_peak_mb": total_estimated_mb,
        "available_mb": total_available_mb,
        "parameters_mb": param_mb,
        "optimizer_states_mb": opt_mb,
        "estimated_activations_mb": activation_mb,
        "recommended_max_batch_size": max(1, int(batch_size * (total_available_mb * 0.85 / max(1.0, total_estimated_mb)))),
    }


def print_gpu_hardware_summary(device: Optional[torch.device] = None) -> None:
    """Print an eye-catching ASCII diagnostic summary of detected compute hardware."""
    tracker = MemoryTracker(device=device)
    info = tracker.get_hardware_info()

    print("=" * 70)
    print("      SCIENTIFIC ML ZERO-BUDGET HARDWARE DIAGNOSTIC REPORT      ")
    print("=" * 70)
    print(f"  Target Device       : {info['device']}")
    print(f"  GPU Hardware Model  : {info['gpu_name']}")
    if info["is_cuda"]:
        print(f"  Total VRAM Available: {info['gpu_total_vram_gb']:.2f} GB")
        print(f"  CUDA Compute Arch   : {info['gpu_compute_capability']}")
        print(f"  Streaming Multiprocs: {info['gpu_multiprocessor_count']}")
    else:
        print("  CUDA Support        : Not Detected (Safe CPU Emulation Active)")
    print(f"  System CPU Cores    : {info['cpu_count_logical']} Logical")
    print(f"  System RAM Available: {info['cpu_ram_available_gb']:.2f} GB / {info['cpu_ram_total_gb']:.2f} GB")
    print("=" * 70)
