"""
Tests for Memmap and Streaming DataLoaders.
"""

from pathlib import Path
import torch
from utils.data_streaming import (
    generate_synthetic_climate_data,
    MemmapScientificDataset,
    StreamingScientificDataset,
    get_optimized_dataloader,
)


def test_memmap_dataset_and_dataloader(tmp_path: Path):
    """Verify zero-RAM Memmap dataset batches load correctly."""
    num_samples = 100
    feat_path, targ_path = generate_synthetic_climate_data(
        output_dir=tmp_path,
        num_samples=num_samples,
        seq_len=6,
        num_stations=4,
        num_features=2,
    )

    dataset = MemmapScientificDataset(
        features_path=feat_path,
        targets_path=targ_path,
        shape_x=(num_samples, 6, 4, 2),
        shape_y=(num_samples, 4),
    )

    assert len(dataset) == 100
    sample_x, sample_y = dataset[0]
    assert sample_x.shape == (6, 4, 2)
    assert sample_y.shape == (4,)
    assert isinstance(sample_x, torch.Tensor)

    # Test DataLoader iteration
    loader = get_optimized_dataloader(dataset, batch_size=16, shuffle=False)
    batch_count = 0
    total_samples = 0
    for bx, by in loader:
        batch_count += 1
        total_samples += bx.size(0)
        assert bx.shape[1:] == (6, 4, 2)
        assert by.shape[1:] == (4,)

    assert batch_count == 7  # 100 // 16 + 1
    assert total_samples == 100


def test_streaming_iterable_dataset(tmp_path: Path):
    """Verify sequential streaming iterable dataset yields correct total samples."""
    num_samples = 64
    feat_path, targ_path = generate_synthetic_climate_data(
        output_dir=tmp_path,
        num_samples=num_samples,
        seq_len=4,
        num_stations=2,
        num_features=2,
    )

    streaming_ds = StreamingScientificDataset(
        features_path=feat_path,
        targets_path=targ_path,
        shape_x=(num_samples, 4, 2, 2),
        shape_y=(num_samples, 2),
        chunk_size=16,
    )

    count = 0
    for x, y in streaming_ds:
        count += 1
        assert x.shape == (4, 2, 2)
        assert y.shape == (2,)

    assert count == 64
