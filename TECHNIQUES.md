# Technical Deep-Dive: Zero-Budget Scientific ML Foundations

A rigorous guide to the mathematical principles, architectural tradeoffs, and hardware execution models governing efficient scientific deep learning on free-tier accelerators (Nvidia T4, Tesla P100).

---

## 1. Automatic Mixed Precision (AMP) & Dynamic Loss Scaling

### The Hardware Reality: Tensor Cores vs CUDA Cores
- **IEEE FP32 (Single Precision):** 1 sign bit, 8 exponent bits, 23 mantissa bits. Dynamic range: $\sim 10^{\pm 38}$.
- **IEEE FP16 (Half Precision):** 1 sign bit, 5 exponent bits, 10 mantissa bits. Dynamic range: $[6 \times 10^{-5}, 65504]$.
- **Bfloat16 (Brain Floating Point):** 1 sign bit, 8 exponent bits, 7 mantissa bits. Dynamic range: $\sim 10^{\pm 38}$.

```
FP32:     [S] [  8 Exponent  ] [        23 Mantissa        ]
FP16:     [S] [ 5 Exp ] [   10 Mantissa   ]
Bfloat16: [S] [  8 Exponent  ] [ 7 Mantissa ]
```

### Why Scientific Models Underflow in FP16
In physical dynamical systems (e.g. Navier-Stokes, stiffness equations, diffusion PDEs), gradient updates can be extremely small ($\ll 10^{-5}$). In plain FP16, these gradients underflow directly to zero:

$$\text{If } |\nabla_\theta \mathcal{L}| < 2^{-14} \approx 6.1 \times 10^{-5} \implies \nabla_\theta \mathcal{L} \to 0$$

### Mathematical Mechanics of `GradScaler`
PyTorch's `GradScaler` solves this by multiplying the loss by an adaptive scale factor $S \gg 1$ before the backward pass:

$$\nabla_\theta \mathcal{L}_{\text{scaled}} = \nabla_\theta (S \cdot \mathcal{L}) = S \cdot \nabla_\theta \mathcal{L}$$

Before the optimizer updates parameters $\theta$:
1. Gradients are unscaled: $\nabla_\theta \mathcal{L} = \frac{1}{S} \cdot \nabla_\theta \mathcal{L}_{\text{scaled}}$
2. If any gradient value is `NaN` or `Inf`, the entire update is skipped, and the scale factor $S$ is halved ($S \leftarrow S \times 0.5$).
3. If $M$ consecutive successful steps occur without overflow, $S$ is doubled ($S \leftarrow S \times 2.0$).

### When to Use / When NOT to Use
- **USE:** On Nvidia Turing (Colab T4), Volta (V100), Ampere (A100), and Hopper (H100) for matrix multiplications and convolutions.
- **DO NOT USE:** For numerical ODE solvers or double-precision scientific physics integrations requiring float64 precision.

---

## 2. Gradient Accumulation & Virtual Batch Sizing

### The Batch Size Constraint on 16GB GPUs
Scientific problems often require large batch sizes ($B \ge 128$) to compute stable Monte Carlo expectations of physical loss functions. However, GPU memory scales with batch size:

$$M_{\text{total}} \approx M_{\text{model}} + M_{\text{optimizer}} + B \times M_{\text{activations\_per\_sample}}$$

### Mathematical Equivalence
Gradient accumulation splits a virtual batch of size $B_{\text{eff}} = B_{\text{micro}} \times K$ across $K$ micro-steps:

$$\nabla_\theta \mathcal{L}_{\text{total}} = \frac{1}{B_{\text{eff}}} \sum_{i=1}^{B_{\text{eff}}} \nabla_\theta \ell_i = \frac{1}{K} \sum_{k=1}^K \left( \frac{1}{B_{\text{micro}}} \sum_{j=1}^{B_{\text{micro}}} \nabla_\theta \ell_{k, j} \right)$$

Dividing each micro-batch loss by $K$ ensures that accumulating gradients over $K$ forward-backward passes yields the exact mathematical gradient of the batch size $B_{\text{eff}}$.

---

## 3. Zero-RAM Data Streaming via Memory-Mapping

### The Colab Host Memory Crash
Google Colab's standard VM allocates only $12.7\text{ GB}$ of host RAM. Attempting to load a $15\text{ GB}$ scientific dataset (e.g. NOAA gridded climate or genomic sequences) triggers an unrecoverable SIGKILL:
`Your session crashed after using all available RAM.`

### How `numpy.memmap` Works Under the Hood
Memory mapping uses the OS virtual memory subsystem (`mmap(2)` syscall). The file is mapped directly into the process address space without reading the bytes into physical RAM.
When PyTorch requests row `idx`:
1. The CPU page-fault handler reads only the $4\text{ KB}$ disk page containing that sample into the kernel file buffer cache.
2. The sample is transferred to GPU memory via DMA (Direct Memory Access).
3. The page is automatically recycled by the OS, maintaining constant $<100\text{ MB}$ RAM footprint throughout the entire training epoch!

---

## 4. MobileNet-Style Depthwise Separable Convolutions

### FLOP & Parameter Comparison
Consider an input tensor with $M$ channels producing $N$ channels with a $D_K \times D_K$ spatial kernel.

- **Standard 2D Convolution:**
  $$\text{Parameters} = D_K \cdot D_K \cdot M \cdot N$$
  $$\text{Computational Cost} = D_K \cdot D_K \cdot M \cdot N \cdot D_F \cdot D_F$$

- **Depthwise Separable Convolution:**
  Consists of a Depthwise Conv ($D_K \times D_K \times M$) followed by a Pointwise Conv ($1 \times 1 \times M \times N$).
  $$\text{Parameters} = D_K \cdot D_K \cdot M + M \cdot N$$
  $$\text{Computational Cost} = D_K \cdot D_K \cdot M \cdot D_F \cdot D_F + M \cdot N \cdot D_F \cdot D_F$$

- **Ratio of Reduction:**
  $$\frac{\text{Cost}_{\text{DS}}}{\text{Cost}_{\text{Standard}}} = \frac{D_K^2 \cdot M + M \cdot N}{D_K^2 \cdot M \cdot N} = \frac{1}{N} + \frac{1}{D_K^2}$$

For a standard $3 \times 3$ kernel ($D_K = 3$) with $N = 128$ output channels:
$$\text{Cost Ratio} = \frac{1}{128} + \frac{1}{9} \approx 0.0078 + 0.111 = 0.1189 \implies \mathbf{88.1\% \text{ reduction in compute!}}$$
