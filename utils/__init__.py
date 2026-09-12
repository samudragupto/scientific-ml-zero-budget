"""
Scientific ML on Zero Budget - Utilities Module
================================================
Comprehensive helper suite for resource-constrained Scientific Machine Learning
on Google Colab and Kaggle free-tier GPUs.
"""

from utils.checkpoint_manager import (
    CheckpointManager,
    load_resilient_checkpoint,
    save_atomic_checkpoint,
    setup_colab_drive_checkpointing,
)
from utils.data_streaming import (
    MemmapScientificDataset,
    StreamingScientificDataset,
    generate_synthetic_climate_data,
    get_optimized_dataloader,
)
from utils.memory_profiler import (
    MemoryTracker,
    benchmark_efficiency,
    estimate_oom_risk,
    print_gpu_hardware_summary,
    profile_memory,
)
from utils.training_utils import (
    AMPTrainer,
    EarlyStopping,
    GradientAccumulator,
    ScientificMetricsTracker,
    create_scientific_lr_scheduler,
)

__all__ = [
    "MemoryTracker",
    "profile_memory",
    "benchmark_efficiency",
    "estimate_oom_risk",
    "print_gpu_hardware_summary",
    "AMPTrainer",
    "GradientAccumulator",
    "create_scientific_lr_scheduler",
    "EarlyStopping",
    "ScientificMetricsTracker",
    "CheckpointManager",
    "save_atomic_checkpoint",
    "load_resilient_checkpoint",
    "setup_colab_drive_checkpointing",
    "MemmapScientificDataset",
    "StreamingScientificDataset",
    "get_optimized_dataloader",
    "generate_synthetic_climate_data",
]
