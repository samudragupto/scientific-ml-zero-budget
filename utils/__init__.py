"""
Scientific ML on Zero Budget - Utilities Module
================================================
Comprehensive helper suite for resource-constrained Scientific Machine Learning
on Google Colab and Kaggle free-tier GPUs.
"""

from utils.memory_profiler import (
    MemoryTracker,
    profile_memory,
    benchmark_efficiency,
    estimate_oom_risk,
    print_gpu_hardware_summary,
)

from utils.training_utils import (
    AMPTrainer,
    GradientAccumulator,
    create_scientific_lr_scheduler,
    EarlyStopping,
    ScientificMetricsTracker,
)

from utils.checkpoint_manager import (
    CheckpointManager,
    save_atomic_checkpoint,
    load_resilient_checkpoint,
    setup_colab_drive_checkpointing,
)

from utils.data_streaming import (
    MemmapScientificDataset,
    StreamingScientificDataset,
    get_optimized_dataloader,
    generate_synthetic_climate_data,
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
