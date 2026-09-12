# Slide Code Snippets & Annotations
## Conference Presentation: Scientific ML on Zero Budget

These code snippets are formatted and annotated specifically for projection slides during the talk.

---

### Slide 1: Automatic Mixed Precision (AMP) on Tensor Cores

```python
import torch

# 1. Enable scaler for FP16 dynamic range compensation
scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

for inputs, targets in dataloader:
    optimizer.zero_grad(set_to_none=True)  # <-- Frees memory instead of writing zeros!

    # 2. Autocast forward pass: Matmuls run in FP16, Reductions run in FP32
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        preds = model(inputs)
        loss = criterion(preds, targets)

    # 3. Scaled backward pass prevents FP16 gradient underflow (near zero)
    scaler.scale(loss).backward()

    # 4. Unscale gradients before clipping, then step optimizer
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
```

> **Speaker Note:** On Google Colab T4 GPUs, FP16 Tensor Cores execute twice as fast as FP32 units and cut activation memory by nearly 50%, with zero scientific accuracy degradation.

---

### Slide 2: Gradient Accumulation (Virtual Enterprise Batches)

```python
accumulation_steps = 4  # Simulates Batch 128 using Batch 32 VRAM footprint!
optimizer.zero_grad(set_to_none=True)

for i, (inputs, targets) in enumerate(dataloader):
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        preds = model(inputs)
        # CRITICAL: Divide loss by accumulation steps so gradients average correctly
        loss = criterion(preds, targets) / accumulation_steps

    scaler.scale(loss).backward()

    # Step optimizer only after accumulating N micro-batches
    if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
```

> **Speaker Note:** Explain why dividing by `accumulation_steps` is mathematically mandatory: otherwise the effective learning rate is multiplied by $K$, causing dynamical models to explode.

---

### Slide 3: Resilient Atomic Checkpointing (Surviving Disconnections)

```python
import os, random, numpy as np, torch

def save_atomic_checkpoint(model, optimizer, epoch, path):
    tmp_path = f"{path}.tmp_{os.getpid()}"
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        # Save RNG states for exact bitwise determinism upon restart
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
    }
    # 1. Write full file to temporary location
    torch.save(state, tmp_path)
    # 2. Atomic filesystem rename prevents corrupted zero-byte files on Google Drive
    os.replace(tmp_path, path)
```

> **Speaker Note:** Colab network drops during a standard `torch.save` leave half-written corrupt files on Google Drive. Atomic replacement solves this completely.

---

### Slide 4: Memory-Mapped Scientific Dataset Streaming

```python
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader

class MemmapScientificDataset(Dataset):
    def __init__(self, bin_path, shape):
        # mode='r': Paged in lazily by OS virtual memory; 0 MB loaded into Python RAM!
        self.data = np.memmap(bin_path, dtype="float32", mode="r", shape=shape)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.data[idx]))

# Colab-Tuned DataLoader: 2 vCPUs, DMA transfers, persistent workers
loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=2,          # Max 2 on Colab to prevent /dev/shm shared memory crashes
    pin_memory=True,        # Enables fast asynchronous CUDA page-locked DMA transfer
    persistent_workers=True # Eliminates worker re-forking overhead between epochs
)
```

> **Speaker Note:** Explain that numpy memmap allows an undergraduate on a free Colab notebook to stream a 50GB ERA5 climate dataset without exceeding Colab's 12.7 GB host RAM limit.

---

### Slide 5: Depthwise Separable Convolutions + Channel Attention

```python
class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        # Depthwise: Spatial filtering per channel (groups = in_ch)
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size, padding=1, groups=in_ch, bias=False)
        # Pointwise: 1x1 linear channel recombination
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return self.bn(self.pw(self.dw(x)))
```

> **Speaker Note:** Standard $3 \times 3$ convolution cost: $D_K^2 \cdot M \cdot N$. Depthwise Separable cost: $D_K^2 \cdot M + M \cdot N$. This is an $8\times$ reduction in arithmetic operations with virtually zero accuracy loss!
