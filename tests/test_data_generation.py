"""
Tests for deterministic synthetic scientific data generation and memmap integrity.
"""

from pathlib import Path
import numpy as np
from utils.data_streaming import generate_synthetic_climate_data


def test_data_generation_shapes_and_values(tmp_path: Path):
    """Verify synthetic climate data conforms to expected dimensions and finite values."""
    num_samples = 200
    seq_len = 12
    num_stations = 4
    num_features = 4

    feat_path, targ_path = generate_synthetic_climate_data(
        output_dir=tmp_path,
        num_samples=num_samples,
        seq_len=seq_len,
        num_stations=num_stations,
        num_features=num_features,
        seed=101,
    )

    assert feat_path.exists()
    assert targ_path.exists()

    # Verify memory map roundtrip
    x = np.memmap(feat_path, dtype="float32", mode="r", shape=(num_samples, seq_len, num_stations, num_features))
    y = np.memmap(targ_path, dtype="float32", mode="r", shape=(num_samples, num_stations))

    assert x.shape == (200, 12, 4, 4)
    assert y.shape == (200, 4)
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(y))

    # Test physical constraints (relative humidity in range 0-100%)
    rh_values = x[..., 1]
    assert np.all(rh_values >= 10.0)
    assert np.all(rh_values <= 100.0)


def test_deterministic_generation(tmp_path: Path):
    """Verify identical random seeds produce bitwise identical dataset files."""
    p1 = tmp_path / "run1"
    p2 = tmp_path / "run2"

    feat1, _ = generate_synthetic_climate_data(output_dir=p1, num_samples=50, seed=42)
    feat2, _ = generate_synthetic_climate_data(output_dir=p2, num_samples=50, seed=42)

    data1 = np.fromfile(feat1, dtype="float32")
    data2 = np.fromfile(feat2, dtype="float32")

    np.testing.assert_array_equal(data1, data2)
