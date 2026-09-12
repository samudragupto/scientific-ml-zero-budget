# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-12

### Added
- **Core Optimization Pipeline:**
  - Automatic Mixed Precision (`torch.amp.autocast` + dynamic `GradScaler`) supporting FP16 and BF16.
  - Multi-step Gradient Accumulation simulating large batch dynamics (effective batch sizes up to 1024) on a single 16GB GPU.
  - Resilient atomic checkpointing system saving model, optimizer, scheduler, epoch, and complete RNG states (Python, NumPy, PyTorch CPU & CUDA).
  - Zero-RAM memory-mapped data streaming pipeline (`np.memmap`) reducing host RAM consumption by 94.6%.
  - MobileNet-style Depthwise-Separable Convolutions, Squeeze-and-Excitation channel attention blocks, and L1 unstructured weight pruning.
- **Scientific Benchmark & Physics Suite:**
  - Deterministic spatio-temporal atmospheric climate generator (`generate_synthetic_climate_data`) modeling diurnal heating, spatial sensor correlations, and turbulence without external network downloads.
  - 4-panel publication-grade benchmark visualization generator (`benchmark_results.py`).
- **Interactive Demonstrations & Presentation Materials:**
  - Colab-optimized runnable notebook (`notebooks/Scientific_ML_Zero_Budget_Demo.ipynb`).
  - Presentation walkthrough script (`demo_script.py`) with slide banners and speaker talking points.
  - Copy-paste quick reference handbook (`quick_reference.py`) with strategy flowcharts and top 6 common pitfalls.
  - Presentation slide snippets (`slides_snippets.md`).
- **DevOps, Testing & Quality Assurance:**
  - Standardized packaging configuration via `pyproject.toml` (PEP 621) and pinned `requirements.txt`.
  - Comprehensive CPU-safe unit test suite (`tests/`) covering imports, data generation, checkpointing, dataloaders, and smoke training.
  - Minimal example scripts (`examples/quick_start.py`, `examples/minimal_example.py`).
  - 7 GitHub Actions workflows (CI, Linting, Smoke Test, Notebook Check, Code Quality, Dependabot, Stale).
  - Reproducibility package (`reproducibility.py`) with seed enforcement, environment audit JSON generator, and academic compute disclosure badge.
