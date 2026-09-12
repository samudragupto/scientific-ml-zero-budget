# Contributing to Scientific ML on Zero Budget

First off, thank you for considering contributing to **Scientific ML on Zero Budget**! We welcome researchers, students, educators, and engineers of all experience levels.

## Core Project Philosophy

Before opening an issue or pull request, please keep our core tenets in mind:
1. **Frugal & Pragmatic:** Efficiency > Enormous. We prioritize methods that run reliably on free-tier compute ($0.00).
2. **Honest About Limitations:** We do not market silver bullets. Trade-offs, diminishing returns, and hard hardware ceilings are treated transparently.
3. **Colab & Kaggle First:** Code should run out of the box with zero external network downloads or complex local drivers.
4. **Reproducible & Clean:** Minimal sufficient complexity. Avoid adding heavy dependencies or unnecessary abstractions.

---

## Ways to Contribute

- **Bug Reports & Fixes:** Identify and fix subtle memory leaks, CUDA OOM edges, or OS incompatibilities.
- **Documentation & Clarifications:** Improve explanations, fix typos, or add intuition to mathematical sections.
- **Scientific Benchmarks:** Report benchmark runs on other accelerators (e.g., Apple Silicon MPS, AMD ROCm, Kaggle T4 x2) with honest numbers.
- **Domain Examples:** Contribute lightweight scientific examples (astrophysics, molecular dynamics, biomedical signals).

---

## Local Development Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/scientific-ml-zero-budget.git
   cd scientific-ml-zero-budget
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   # Linux / macOS:
   source .venv/bin/activate
   # Windows:
   .venv\Scripts\activate
   ```

3. **Install runtime and development dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   pip install -e .
   ```

4. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

---

## Running Tests & Code Quality

Always verify your changes before submitting a PR:

```bash
# Run automated tests
pytest

# Run fast CPU smoke test of main pipeline
python main_demo.py

# Check linting and formatting with Ruff
ruff check .
ruff format --check .

# Or run everything via Makefile:
make check
```

---

## Pull Request Guidelines

1. **Keep PRs Focused:** One feature, fix, or documentation enhancement per pull request.
2. **Preserve Educational Clarity:** Add comments explaining *why* an optimization is necessary (e.g., memory saving, DMA transfer, page-locking).
3. **Avoid Feature Bloat:** Ask yourself: *"Does an undergraduate student on a free Colab account benefit from this without additional setup friction?"*
4. **Fill Out the PR Template:** Describe your motivation, the testing performed, and any hardware-dependent observations.
