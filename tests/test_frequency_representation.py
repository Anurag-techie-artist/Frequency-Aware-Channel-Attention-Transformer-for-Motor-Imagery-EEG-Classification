"""
Unit Tests for Frequency-Aware EEG Representation Module (Phase 3).

Tests:
- Configuration validation (low < high, high < Nyquist bounds)
- Shape transformations for single window (Bands, Channels, Samples) and batch (N, Bands, Channels, Samples)
- Preservation of channel and sample dimensions
- Deterministic sub-band ordering
- Absence of NaN or Inf values
- Backward compatibility (representation="time" vs representation="frequency")
- Debug artifact exports (frequency_tensor.npy, frequency_metadata.json, frequency_summary.json)
"""

import os
import sys
import unittest
import json
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import gc
from datasets.transforms.frequency import (
    FrequencyBandConfig,
    FrequencyRepresentationConfig,
    FrequencyRepresentation,
)
from datasets.pipeline import EEGPreprocessingPipeline
from datasets.dataset import HGDDataset


class TestFrequencyRepresentation(unittest.TestCase):
    """Test suite for FrequencyRepresentation module and integration."""

    def setUp(self):
        self.sample_edf = os.path.join(PROJECT_ROOT, "hgd", "train1", "1.edf")
        self.fs = 250.0
        self.n_channels = 133
        self.n_samples = 250

        # Synthetic dummy window (Channels, Samples)
        np.random.seed(42)
        self.dummy_window = np.random.randn(self.n_channels, self.n_samples).astype(np.float32)
        # Synthetic dummy batch (N, Channels, Samples)
        self.dummy_batch = np.random.randn(5, self.n_channels, self.n_samples).astype(np.float32)

    def tearDown(self):
        gc.collect()

    def test_config_validation_valid(self):
        """Test that valid configuration passes validation."""
        config = FrequencyRepresentationConfig(sampling_rate=250.0)
        freq_rep = FrequencyRepresentation(config=config)
        self.assertEqual(len(freq_rep.config.bands), 4)

    def test_config_validation_invalid_low_high(self):
        """Test that low >= high raises ValueError."""
        invalid_bands = [FrequencyBandConfig(name="invalid", low=30.0, high=20.0)]
        config = FrequencyRepresentationConfig(sampling_rate=250.0, bands=invalid_bands)
        with self.assertRaises(ValueError):
            FrequencyRepresentation(config=config)

    def test_config_validation_nyquist_exceeded(self):
        """Test that high >= Nyquist (fs / 2) raises ValueError."""
        invalid_bands = [FrequencyBandConfig(name="invalid", low=100.0, high=130.0)]
        config = FrequencyRepresentationConfig(sampling_rate=250.0, bands=invalid_bands)
        with self.assertRaises(ValueError):
            FrequencyRepresentation(config=config)

    def test_single_window_shape(self):
        """Test single window transformation: (Channels, Samples) -> (Bands, Channels, Samples)."""
        freq_rep = FrequencyRepresentation()
        out_tensor, metadata = freq_rep.extract(self.dummy_window)

        expected_shape = (4, self.n_channels, self.n_samples)
        self.assertEqual(out_tensor.shape, expected_shape)
        self.assertEqual(metadata.tensor_shape, list(expected_shape))
        self.assertEqual(metadata.execution_time_seconds > 0, True)

    def test_batch_windows_shape(self):
        """Test batch transformation: (N, Channels, Samples) -> (N, Bands, Channels, Samples)."""
        freq_rep = FrequencyRepresentation()
        out_tensor, metadata = freq_rep.extract(self.dummy_batch)

        expected_shape = (5, 4, self.n_channels, self.n_samples)
        self.assertEqual(out_tensor.shape, expected_shape)
        self.assertEqual(metadata.tensor_shape, list(expected_shape))

    def test_channel_and_sample_dimension_preserved(self):
        """Test that channel count (133) and sample count (250) are preserved across all sub-bands."""
        freq_rep = FrequencyRepresentation()
        out_tensor, _ = freq_rep.extract(self.dummy_window)

        # Check for each sub-band slice
        for b_idx in range(4):
            band_slice = out_tensor[b_idx]
            self.assertEqual(band_slice.shape, (self.n_channels, self.n_samples))

    def test_deterministic_band_ordering(self):
        """Test that sub-band ordering is deterministic (0: theta, 1: alpha, 2: beta, 3: gamma)."""
        freq_rep = FrequencyRepresentation()
        _, metadata = freq_rep.extract(self.dummy_window)

        band_names = [b["name"] for b in metadata.frequency_bands]
        self.assertEqual(band_names, ["theta", "alpha", "beta", "gamma"])

    def test_no_nan_or_inf(self):
        """Test that extracted frequency tensor contains no NaN or Inf values."""
        freq_rep = FrequencyRepresentation()
        out_tensor, _ = freq_rep.extract(self.dummy_window)

        self.assertFalse(np.isnan(out_tensor).any(), "NaN values found in frequency tensor!")
        self.assertFalse(np.isinf(out_tensor).any(), "Inf values found in frequency tensor!")

    def test_pipeline_representation_modes(self):
        """Test pipeline process with representation='time' vs representation='frequency'."""
        if not os.path.exists(self.sample_edf):
            self.skipTest(f"Sample EDF file not found at {self.sample_edf}")

        pipeline = EEGPreprocessingPipeline()

        # 1. Time domain representation
        X_time, y_time, _ = pipeline.process(self.sample_edf, representation="time")
        self.assertEqual(X_time.ndim, 3)  # (N_windows, Channels, Samples)

        # 2. Frequency domain representation
        X_freq, y_freq, _ = pipeline.process(self.sample_edf, representation="frequency")
        self.assertEqual(X_freq.ndim, 4)  # (N_windows, Bands, Channels, Samples)

        self.assertEqual(X_freq.shape[0], X_time.shape[0])
        self.assertEqual(X_freq.shape[1], 4)  # 4 sub-bands
        self.assertEqual(X_freq.shape[2], X_time.shape[1])
        self.assertEqual(X_freq.shape[3], X_time.shape[2])

    def test_dataset_representation_modes(self):
        """Test PyTorch HGDDataset with representation='time' vs representation='frequency'."""
        if not os.path.exists(self.sample_edf):
            self.skipTest(f"Sample EDF file not found at {self.sample_edf}")

        # 1. Time representation dataset
        ds_time = HGDDataset(self.sample_edf, representation="time")
        sample_x_time, _ = ds_time[0]
        self.assertEqual(sample_x_time.ndim, 2)  # (Channels, Samples)

        # 2. Frequency representation dataset
        ds_freq = HGDDataset(self.sample_edf, representation="frequency")
        sample_x_freq, _ = ds_freq[0]
        self.assertEqual(sample_x_freq.ndim, 3)  # (Bands, Channels, Samples)
        self.assertEqual(sample_x_freq.shape[0], 4)  # 4 sub-bands

    def test_debug_export(self):
        """Test generation of frequency_tensor.npy, frequency_metadata.json, and frequency_summary.json."""
        if not os.path.exists(self.sample_edf):
            self.skipTest(f"Sample EDF file not found at {self.sample_edf}")

        pipeline = EEGPreprocessingPipeline()
        debug_res = pipeline.process_debug(self.sample_edf, representation="frequency")

        debug_dir = os.path.join(PROJECT_ROOT, "outputs", "debug")
        tensor_path = os.path.join(debug_dir, "frequency_tensor.npy")
        metadata_path = os.path.join(debug_dir, "frequency_metadata.json")
        summary_path = os.path.join(debug_dir, "frequency_summary.json")

        self.assertTrue(os.path.exists(tensor_path), f"File {tensor_path} does not exist!")
        self.assertTrue(os.path.exists(metadata_path), f"File {metadata_path} does not exist!")
        self.assertTrue(os.path.exists(summary_path), f"File {summary_path} does not exist!")

        with open(summary_path, "r", encoding="utf-8") as f:
            summary_data = json.load(f)

        self.assertEqual(summary_data["bands"], 4)
        self.assertEqual(summary_data["band_names"], ["theta", "alpha", "beta", "gamma"])
        print("\n[PASSED] FrequencyRepresentation unit tests completed successfully!")


if __name__ == "__main__":
    unittest.main()
