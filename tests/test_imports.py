"""
Tests for clean module imports and zero-dependency package initialization.
"""


def test_imports_utils():
    """Verify all utilities import cleanly without runtime side effects."""
    from utils import (
        AMPTrainer,
        CheckpointManager,
        MemmapScientificDataset,
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

    assert hasattr(main_demo, "BaselineHeavyScientificModel")
    assert hasattr(main_demo, "EfficientScientificModel")
    assert hasattr(benchmark_results, "get_hardware_benchmark_data")
    assert hasattr(reproducibility, "seed_everything")
    assert hasattr(quick_reference, "recommend_strategy")
