"""
Scientific ML on Zero Budget: Quick Reference Guide
===================================================
A concise, copy-paste ready reference handbook containing drop-in code snippets,
algorithmic decision flowcharts, and a comprehensive catalog of common free-tier
GPU pitfalls and solutions.
"""

from __future__ import annotations

from typing import Any, Dict


# ==============================================================================
# 1. DROP-IN CODE SNIPPETS
# ==============================================================================

SNIPPET_AMP = '''
# ------------------------------------------------------------------------------
# SNIPPET 1: Automatic Mixed Precision (AMP) with GradScaler
# Memory reduction: ~40-50% | Speedup: ~1.8x - 2.5x on Nvidia T4 Tensor Cores
# ------------------------------------------------------------------------------
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Enable scaler for float16 on CUDA; disable for bfloat16 or CPU
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

for epoch in range(num_epochs):
    model.train()
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # set_to_none=True saves memory!

        # Autocast runs forward ops in FP16, reductions in FP32
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=torch.cuda.is_available()):
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

        # Scales loss before backprop to prevent FP16 gradient underflow
        scaler.scale(loss).backward()

        # Unscales gradients, applies clipping, and steps optimizer
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
'''

SNIPPET_GRAD_ACCUM = '''
# ------------------------------------------------------------------------------
# SNIPPET 2: Gradient Accumulation (Simulate Large Batch on 16GB GPU)
# Memory required: Batch 32 | Effective Batch Size: 32 x 8 = 256
# ------------------------------------------------------------------------------
import torch

accumulation_steps = 8
optimizer.zero_grad(set_to_none=True)

for i, (inputs, targets) in enumerate(dataloader):
    inputs, targets = inputs.to(device), targets.to(device)

    with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
        preds = model(inputs)
        # CRUCIAL: Divide loss by accumulation steps so gradients average correctly
        loss = criterion(preds, targets) / accumulation_steps

    scaler.scale(loss).backward()

    # Step optimizer only after accumulating enough micro-batches
    if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
'''

SNIPPET_ATOMIC_CHECKPOINT = '''
# ------------------------------------------------------------------------------
# SNIPPET 3: Atomic Resilient Checkpointing (Safe against Colab Disconnects)
# Writes to .tmp first, then renames atomically so .pt file is NEVER corrupted
# ------------------------------------------------------------------------------
import os
import random
import numpy as np
import torch

def save_resilient_checkpoint(model, optimizer, epoch, path):
    tmp_path = f"{path}.tmp_{os.getpid()}"
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
    }
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)  # Atomic operation
'''

SNIPPET_MEMMAP_DATASET = '''
# ------------------------------------------------------------------------------
# SNIPPET 4: Memory-Mapped Scientific Dataset (Zero RAM Footprint)
# Stream 50GB climate/spectroscopy matrices from disk with <100MB RAM
# ------------------------------------------------------------------------------
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class ZeroRamDataset(Dataset):
    def __init__(self, dat_filepath, shape, dtype="float32"):
        # mode='r' means memory is mapped lazily from disk; no array allocation in RAM
        self.data = np.memmap(dat_filepath, dtype=dtype, mode="r", shape=shape)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # Only the requested row is read into memory
        return torch.from_numpy(np.array(self.data[idx]))

# Recommended Colab DataLoader settings (2 vCPUs)
loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=2,         # Never > 2 on Colab (prevents /dev/shm crashes)
    pin_memory=True,       # Fast asynchronous DMA transfer to GPU
    persistent_workers=True
)
'''


# ==============================================================================
# 2. DECISION FLOWCHART
# ==============================================================================

DECISION_FLOWCHART_ASCII = """
+-----------------------------------------------------------------------------+
|              SCIENTIFIC ML ZERO-BUDGET OPTIMIZATION FLOWCHART               |
+-----------------------------------------------------------------------------+

                           [START SCIENTIFIC ML PROJECT]
                                         |
                                         v
                      Is dataset size > 60% of Host RAM (e.g. > 8 GB)?
                                    /        \\
                           YES     /          \\    NO
                                  v            v
               [Use np.memmap or Iterable]   [Standard in-memory TensorDataset]
               [Set num_workers=2, pin=True]
                                  \\            /
                                   \\          /
                                    v        v
                           Does Model fit in GPU VRAM (16GB T4)?
                                    /        \\
                           YES     /          \\    NO
                                  v            v
                      Can we run desired      [Apply PyTorch AMP (FP16)]
                      effective batch size?   [Gradient Accumulation: 4-8 steps]
                            /       \\         [Gradient Checkpointing]
                   YES     /         \\  NO     |
                          v           v        v
           [Enable AMP (FP16)]    [Simulate batch with Gradient Accumulation]
           [optimizer.zero_grad(set_to_none=True)]
                          |
                          v
           Long-running training (> 30 minutes on Colab / Kaggle)?
                            /       \\
                   YES     /         \\  NO
                          v           v
           [Mount Google Drive to /content/drive]      [Local Checkpoints]
           [Atomic checkpointing + RNG saving]
           [Register SIGTERM signal handler]
                          |
                          v
           Model inference too slow / deployment budget zero?
                            /       \\
                   YES     /         \\  NO
                          v           v
           [Replace dense Convs with MobileNet Depthwise Separable Convs]
           [Add Squeeze-and-Excitation attention channel reduction]
           [Apply L1 Unstructured Pruning (30-50% sparsity)]
                          |
                          v
                     [DEPLOYED AT ZERO COST!]
"""


def recommend_strategy(
    dataset_size_gb: float,
    gpu_vram_gb: float = 15.0,
    model_param_count_millions: float = 25.0,
    host_ram_gb: float = 12.0,
) -> Dict[str, Any]:
    """Provide algorithmic recommendations based on hardware and workload constraints.

    Args:
        dataset_size_gb: Estimated dataset size on disk in GB.
        gpu_vram_gb: Available GPU memory in GB (Colab T4 ~ 15.0 GB).
        model_param_count_millions: Model parameter count in millions.
        host_ram_gb: System RAM (Colab free tier ~ 12.7 GB).

    Returns:
        Dictionary of recommended techniques and configuration parameters.
    """
    recs: Dict[str, Any] = {}

    # Data strategy
    if dataset_size_gb > (host_ram_gb * 0.5):
        recs["data_loading"] = "MemmapScientificDataset (np.memmap mode='r')"
        recs["num_workers"] = 2
        recs["data_rationale"] = (
            f"Dataset ({dataset_size_gb:.1f} GB) risks crashing host RAM ({host_ram_gb:.1f} GB). "
            "Memory-mapped zero-copy access pages tensors directly from disk on-demand."
        )
    else:
        recs["data_loading"] = "In-memory TensorDataset or cached DataLoader"
        recs["num_workers"] = 2
        recs["data_rationale"] = "Dataset fits comfortably within host RAM."

    # Precision & batch strategy
    recs["precision"] = "Automatic Mixed Precision (torch.amp.autocast, dtype=float16)"
    if model_param_count_millions > 40.0:
        recs["batch_strategy"] = "Micro-batch size 16 + Gradient Accumulation steps 8"
        recs["effective_batch_size"] = 128
        recs["activation_checkpointing"] = "Recommended (torch.utils.checkpoint)"
    elif model_param_count_millions > 10.0:
        recs["batch_strategy"] = "Micro-batch size 32 + Gradient Accumulation steps 4"
        recs["effective_batch_size"] = 128
        recs["activation_checkpointing"] = "Optional"
    else:
        recs["batch_strategy"] = "Native batch size 64 or 128 with AMP"
        recs["effective_batch_size"] = 64
        recs["activation_checkpointing"] = "Not needed"

    # Checkpoint strategy
    recs["checkpointing"] = "Atomic save to Google Drive (/content/drive/MyDrive) with RNG capture"
    recs["storage_pruning"] = "Keep top 3 checkpoints (prevent Google Drive / Kaggle quota overflow)"

    return recs


# ==============================================================================
# 3. COMMON PITFALLS CATALOG
# ==============================================================================

PITFALLS_CATALOG = [
    {
        "id": "PITFALL-01",
        "title": "Accumulating Computation Graph in Loss Tracking",
        "severity": "CRITICAL (Causes OOM by epoch 2)",
        "error_message": "CUDA out of memory. Tried to allocate 256.00 MiB (GPU 0; 15.78 GiB total capacity)",
        "bad_code": "total_loss += loss  # Keeps full backprop graph of entire epoch in VRAM!",
        "good_code": "total_loss += loss.item() * batch_size  # Extracts scalar float, freeing graph",
        "why": "Assigning the raw tensor keeps PyTorch's dynamic autograd graph alive across all batches, leaking gigabytes of memory.",
    },
    {
        "id": "PITFALL-02",
        "title": "Missing `scaler.update()` or Unscaling Multiple Times in AMP",
        "severity": "HIGH (Causes training divergence or NaNs)",
        "error_message": "RuntimeError: unscale_() has already been called on this optimizer since the last update().",
        "bad_code": "scaler.step(optimizer) # Forgot scaler.update()!",
        "good_code": "scaler.step(optimizer)\nscaler.update() # Adjusts scale factor dynamically to avoid NaNs",
        "why": "GradScaler dynamically monitors gradient values. If you forget to update the scale factor, gradients will explode or vanish.",
    },
    {
        "id": "PITFALL-03",
        "title": "Excessive DataLoader Workers in Colab/Kaggle",
        "severity": "CRITICAL (Silent kernel crash / Session restart)",
        "error_message": "RuntimeError: DataLoader worker (pid 1234) is killed by signal: Bus error / Your session crashed.",
        "bad_code": "DataLoader(dataset, batch_size=64, num_workers=8) # Colab free tier only has 2 vCPUs!",
        "good_code": "DataLoader(dataset, batch_size=64, num_workers=2, pin_memory=True, persistent_workers=True)",
        "why": "Colab limits shared memory (/dev/shm) to 5-10GB. Forking 8 workers creates 8 IPC tensor queues, causing an instant OS bus error.",
    },
    {
        "id": "PITFALL-04",
        "title": "Non-Atomic Checkpointing on Google Drive",
        "severity": "HIGH (Corrupted unrecoverable weights after timeout)",
        "error_message": "RuntimeError: PytorchStreamReader failed reading zip archive: failed finding central directory",
        "bad_code": "torch.save(checkpoint, '/content/drive/MyDrive/model.pt') # Broken if disconnected mid-write!",
        "good_code": "torch.save(checkpoint, 'tmp.pt')\nos.replace('tmp.pt', '/content/drive/MyDrive/model.pt') # Atomic",
        "why": "Network latency to Google Drive means multi-hundred MB writes take several seconds. If disconnected during write, the file is corrupted.",
    },
    {
        "id": "PITFALL-05",
        "title": "Inner-Loop CPU Synchronization Stalls",
        "severity": "MEDIUM (Slows training by 3x - 5x)",
        "error_message": "No crash, but GPU utilization remains below 25%",
        "bad_code": "for x, y in loader:\n    loss = model(x)\n    print(f'Batch loss: {loss.item()}') # Forces CPU sync every batch!",
        "good_code": "# Accumulate asynchronously and print once per N batches or epoch\nif step % 50 == 0:\n    print(loss.item())",
        "why": "Calling .item() or .cpu().numpy() stalls the CUDA execution stream, forcing the CPU to block waiting for the GPU.",
    },
    {
        "id": "PITFALL-06",
        "title": "Neglecting `optimizer.zero_grad(set_to_none=True)`",
        "severity": "LOW-MEDIUM (Wasteful memory allocation)",
        "error_message": "Unnecessary 5-10% VRAM overhead",
        "bad_code": "optimizer.zero_grad() # Writes zeros into existing gradient tensor buffers",
        "good_code": "optimizer.zero_grad(set_to_none=True) # Deallocates buffer, freeing memory for activations",
        "why": "Setting gradients to None allows the PyTorch caching memory allocator to repurpose buffer memory for subsequent layers.",
    },
]


def print_quick_reference() -> None:
    """Print the complete quick reference guide in terminal."""
    print("=" * 80)
    print("         SCIENTIFIC ML ON ZERO BUDGET: QUICK REFERENCE HANDBOOK         ")
    print("=" * 80)
    print(DECISION_FLOWCHART_ASCII)
    print("\n" + "=" * 80)
    print("                     TOP 6 COMMON PITFALLS & SOLUTIONS                  ")
    print("=" * 80)
    for p in PITFALLS_CATALOG:
        print(f"\n[{p['id']}] {p['title']} ({p['severity']})")
        print(f"  Error    : {p['error_message']}")
        print(f"  Fix      : {p['good_code'].splitlines()[0]}")
        print(f"  Why      : {p['why']}")


if __name__ == "__main__":
    print_quick_reference()
    sample_recs = recommend_strategy(dataset_size_gb=14.0, model_param_count_millions=30.0)
    print("\n--- Strategy Recommendation Example ---")
    for k, v in sample_recs.items():
        print(f"  {k:20s}: {v}")
