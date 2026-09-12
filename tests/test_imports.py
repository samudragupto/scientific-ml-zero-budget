"""
Tests for clean module imports and zero-dependency package initialization.
"""

def test_imports_utils():
    """Verify all utilities import cleanly without runtime side effects."""
    from utils import (
        AMPTrainer,
        CheckpointManager,
        EarlyStopping,
        GradientAccumulator,
        MemmapScientificDataset,
        MemoryTracker,
        ScientificMetricsTracker,
        StreamingScientificDataset,
        benchmark_efficiency,
        estimate_oom_risk,
        generate_synthetic_climate_data,
        get_optimized_dataloader,
        load_resilient_checkpoint,
        print_gpu_hardware_summary,
        profile_memory,
        save_atomic_checkpoint,
        setup_colab_drive_checkpointing,
    )
    assert AMPTrainer is not None
    assert CheckpointManager is not None
    assert MemmapScientificDataset is not None


def test_imports_core_scripts():
    """Verify core executable modules import without throwing errors."""
    import benchmark_results
    import main_demo
    import quick_reference
    import reproducibility
    import setup_colab

    assert hasattr(main_demo, "BaselineHeavyScientificModel")
    assert hasattr(main_demo, "EfficientScientificModel")
    assert hasattr(benchmark_results, "get_hardware_benchmark_data")
    assert hasattr(reproducibility, "seed_everything")
    assert hasattr(quick_reference, "recommend_strategy")
