# Troubleshooting & Survival Guide for Free Colab & Kaggle GPUs

This handbook provides exact root causes, error messages, and verified solutions for the most frustrating failures encountered when training scientific deep learning models on free cloud GPU tiers.

---

## 1. Google Colab Session Disconnected / Runtime Reset

### Symptom
- Browser displays: *"Cannot connect to GPU backend"* or *"Your session was disconnected because of inactivity or session limit (12 hours)."*
- Local runtime files inside `/content/` are wiped out completely.

### Root Cause
Google Colab enforces a hard 12-hour session lifetime and disconnects users after 30-60 minutes of inactive browser interaction. Any checkpoint saved directly to `/content/` or `/tmp/` is permanently destroyed.

### Solution
1. **Always mount Google Drive at notebook start:**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
2. **Use the `CheckpointManager` from `utils/checkpoint_manager.py`:**
   It writes atomically to `/content/drive/MyDrive/SciML_Checkpoints/`.
3. **Colab Keep-Alive Trick (Browser Console):**
   Press `Ctrl+Shift+I` (DevTools) -> Console tab -> paste:
   ```javascript
   function ClickConnect(){
     console.log("Keeping Colab Alive...");
     document.querySelector("colab-connect-button")?.shadowRoot?.querySelector("#connect")?.click();
   }
   setInterval(ClickConnect, 60000);
   ```

---

## 2. CUDA Out of Memory (OOM) Errors

### Symptom
```
RuntimeError: CUDA out of memory. Tried to allocate 512.00 MiB (GPU 0; 15.78 GiB total capacity; 14.92 GiB already allocated)
```

### Root Cause
1. **Batch size is too large** for the 15-16GB VRAM limit.
2. **Autograd graph leakage:** Writing `total_loss += loss` retains the computation graph across all batches.
3. **PyTorch caching allocator fragmentation:** Memory allocated by earlier operations is reserved but fragmented.

### Solution Checklist
- [x] Change `total_loss += loss` to `total_loss += loss.item() * batch_size`.
- [x] Use `optimizer.zero_grad(set_to_none=True)` to release gradient buffers rather than filling them with zeros.
- [x] Reduce micro-batch size from 64 to 16, and use `GradientAccumulator(accumulation_steps=4)` to preserve your effective batch size.
- [x] Enable Automatic Mixed Precision (AMP):
  ```python
  with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
      outputs = model(inputs)
  ```
- [x] Flush cache if recovering:
  ```python
  import gc, torch
  gc.collect()
  torch.cuda.empty_cache()
  ```

---

## 3. Slow Data Loading & GPU Starvation

### Symptom
- `nvidia-smi` reports GPU Utilization under 20%.
- Training is bottlenecked by CPU data preprocessing.

### Root Cause
- Too many or too few `DataLoader` workers (`num_workers=8` causes bus errors on Colab's 2-vCPU VMs; `num_workers=0` forces serial CPU-GPU waiting).
- Loading individual small files from Google Drive (Google Drive FUSE has catastrophic multi-file I/O latency).

### Solution
1. **Never load thousands of individual CSVs or images directly from Google Drive.** Copy a single `.zip` or `.tar` archive to local `/content/` disk first and extract, or use binary memory-mapped files:
   ```python
   # Load from single contiguous binary memmap
   dataset = MemmapScientificDataset("data.dat", shape=(10000, 24, 8, 4))
   ```
2. **Optimal Colab DataLoader configuration:**
   ```python
   loader = DataLoader(
       dataset,
       batch_size=64,
       num_workers=2,          # Exactly matches Colab's 2 vCPUs
       pin_memory=True,        # Enables fast asynchronous CUDA page-locked DMA
       persistent_workers=True # Avoids re-forking workers every epoch
   )
   ```

---

## 4. Numerical Instability & Loss Becoming `NaN` in FP16

### Symptom
Validation loss suddenly prints `nan` after epoch 3 or 4.

### Root Cause
- In half-precision (FP16), maximum representable number is 65,504. Stiff physical equations or large learning rates cause exponential gradients that overflow into `+Inf` or `NaN`.
- Missing `GradScaler` or uncalibrated learning rates.

### Solution
1. **Always wrap FP16 with `torch.amp.GradScaler`:**
   ```python
   scaler = torch.amp.GradScaler("cuda", enabled=True)
   scaler.scale(loss).backward()
   scaler.unscale_(optimizer)
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Clip gradients!
   scaler.step(optimizer)
   scaler.update()
   ```
2. **Apply Learning Rate Warmup:**
   Initial steps without warmup cause turbulent parameter spikes. Use `create_scientific_lr_scheduler(optimizer, warmup_steps=100)`.
3. **If problem is stiff ODE/PDE:** Consider Bfloat16 (`dtype=torch.bfloat16`) if running on Ampere+ GPUs, which has the same dynamic range as FP32.

---

## 5. Corrupted Checkpoint Files

### Symptom
```
RuntimeError: PytorchStreamReader failed reading zip archive: failed finding central directory
```

### Root Cause
Google Colab timed out or was terminated in the middle of a `torch.save()` write to Google Drive. The file on Drive is half-written and permanently corrupted.

### Solution
Use **atomic saving**: write to a temporary file on local disk first, then execute an atomic rename:
```python
tmp_path = f"{target_path}.tmp"
torch.save(payload, tmp_path)
os.replace(tmp_path, target_path) # Atomic on POSIX and Windows filesystems
```
The `CheckpointManager` class handles this automatically.
