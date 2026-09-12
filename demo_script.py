"""
Live Presentation Demonstration Script
======================================
Interactive console walkthrough script designed for the live conference presentation:
"Scientific ML on Zero Budget: Training Real Models with Free Colab/Kaggle GPUs".

Includes step-by-step speaker narration pauses, performance metric callouts,
and hardware profiling highlights.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from main_demo import run_scientific_ml_demonstration


def print_speaker_banner(step_num: int, title: str, talking_points: list[str]) -> None:
    """Print an eye-catching presentation banner with speaker talking points."""
    print("\n" + "=" * 80)
    print(f"  [SLIDE {step_num}] {title.upper()}")
    print("=" * 80)
    for point in talking_points:
        print(f"  * {point}")
    print("-" * 80)
    time.sleep(0.5)


def run_live_presentation_demo() -> None:
    """Run through the conference presentation talk flow."""
    print("=" * 80)
    print("      SciPy India: Scientific ML on Zero Budget (Live Demo Walkthrough)    ")
    print("=" * 80)

    print_speaker_banner(
        step_num=1,
        title="The Zero Budget Constraint: Free T4 & P100 Realities",
        talking_points=[
            "Free Colab gives you a single 15-16GB GPU with only 12GB Host RAM and 2 vCPUs.",
            "Long running jobs face 12-hour session terminations and unexpected preemption.",
            "Naive deep learning scripts crash with CUDA OOM or host memory exhaustion.",
        ],
    )

    print_speaker_banner(
        step_num=2,
        title="The Scientific Target: Spatio-Temporal Climate Forecasting",
        talking_points=[
            "Real physical dynamics: Fourier diurnal heating, spatial sensor correlations.",
            "Zero external download dependencies: deterministic disk-backed memmap synthesis.",
            "Benchmarking Baseline vs AMP vs Gradient Accumulation vs Efficient Architecture.",
        ],
    )

    # Execute demonstration pipeline
    results = run_scientific_ml_demonstration(num_samples=1000, epochs_per_phase=1, batch_size=32)

    print_speaker_banner(
        step_num=3,
        title="Key Takeaways for Attendees",
        talking_points=[
            "1. Mixed Precision (AMP): ~50% VRAM saved + >2x speedup on Tensor Cores with zero loss in MSE.",
            "2. Gradient Accumulation: Simulate batch sizes up to 1024 on a free 16GB GPU.",
            "3. Memory-Mapping: Stream 50GB scientific datasets using less than 100MB of Host RAM.",
            "4. Atomic Checkpoints: Never lose progress to Colab 12-hour timeouts again.",
            "5. Efficient Convolutions: MobileNet-style Depthwise-Separable convs cut parameters by ~80%.",
        ],
    )
    print("\n[Conference Demo Complete] Ready for audience Q&A!")


if __name__ == "__main__":
    run_live_presentation_demo()
