"""
Unit Tests for v0.11.0 Trial-Level Lazy Dataset Cache Architecture.
"""

import os
import sys
import tempfile
import unittest
import random
from unittest.mock import patch, MagicMock

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.cache import CacheManager, FileLRUCache, compute_config_hash, CACHE_VERSION
from datasets.dataset import HGDDataset
from datasets.pipeline import EEGPreprocessingPipeline, PreprocessingConfig
from datasets.windowing import generate_sliding_windows, extract_single_window_from_trial


class TestTrialCache(unittest.TestCase):
    """Comprehensive test suite for v0.11.0 trial-level lazy dataset cache architecture."""

    @patch.object(EEGPreprocessingPipeline, "process_trials")
    def test_trial_cache_build_and_lazy_getitem(self, mock_process_trials):
        """Test trial-level cache creation and lazy window extraction inside __getitem__()."""
        # Mock 5 trials of shape (5, 4, 133, 751)
        raw_trials = np.random.randn(5, 4, 133, 751).astype(np.float32)
        raw_labels = np.array([0, 1, 2, 3, 0], dtype=np.int64)
        raw_tids = np.arange(5, dtype=np.int64)
        mock_process_trials.return_value = (raw_trials, raw_labels, raw_tids)

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_cfg = {"enabled": True, "directory": tmp_dir, "max_open_cache_files": 2}
            pipeline = EEGPreprocessingPipeline()

            ds = HGDDataset(
                file_paths=["/fake/path/sample.edf"],
                pipeline=pipeline,
                representation="frequency",
                cache_config=cache_cfg,
            )

            # Each trial of length 751 with win=250, stride=50 yields 11 windows -> 5 * 11 = 55 windows
            self.assertEqual(len(ds), 55)

            # Test item retrieval
            x_sample, y_sample = ds[0]
            self.assertEqual(x_sample.shape, torch.Size([4, 133, 250]))
            self.assertEqual(y_sample.item(), 0)

    @patch.object(EEGPreprocessingPipeline, "process_trials")
    def test_random_equivalence_between_eager_and_lazy(self, mock_process_trials):
        """Automated Equivalence Test: Assert x_old == x_new and y_old == y_new for random sample indices."""
        raw_trials = np.random.randn(8, 4, 133, 751).astype(np.float32)
        raw_labels = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
        raw_tids = np.arange(8, dtype=np.int64)
        mock_process_trials.return_value = (raw_trials, raw_labels, raw_tids)

        # Compute eager reference windows
        eager_x, eager_y, _ = generate_sliding_windows(raw_trials, raw_labels, window_size=250, stride=50, normalize=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_cfg = {"enabled": True, "directory": tmp_dir}
            pipeline = EEGPreprocessingPipeline()

            ds = HGDDataset(
                file_paths=["/fake/sample.edf"],
                pipeline=pipeline,
                representation="frequency",
                cache_config=cache_cfg,
            )

            self.assertEqual(len(ds), len(eager_x))

            # Sample random indices and verify equivalence
            sample_indices = random.sample(range(len(ds)), min(30, len(ds)))
            for idx in sample_indices:
                x_lazy, y_lazy = ds[idx]
                x_eager = torch.from_numpy(eager_x[idx])

                torch.testing.assert_close(x_lazy, x_eager, atol=1e-5, rtol=1e-5)
                self.assertEqual(y_lazy.item(), eager_y[idx])

    @patch.object(EEGPreprocessingPipeline, "process_trials")
    def test_lru_cache_eviction(self, mock_process_trials):
        """Test FileLRUCache capacity and eviction behavior with trial files."""
        mock_process_trials.return_value = (
            np.random.randn(3, 133, 751).astype(np.float32),
            np.zeros(3, dtype=np.int64),
            np.zeros(3, dtype=np.int64),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            lru = FileLRUCache(max_open_cache_files=2)
            f1 = os.path.join(tmp_dir, "f1_trials.pt")
            f2 = os.path.join(tmp_dir, "f2_trials.pt")
            f3 = os.path.join(tmp_dir, "f3_trials.pt")

            torch.save({"trials": torch.randn(3, 133, 751), "labels": torch.zeros(3), "cache_version": CACHE_VERSION}, f1)
            torch.save({"trials": torch.randn(3, 133, 751), "labels": torch.zeros(3), "cache_version": CACHE_VERSION}, f2)
            torch.save({"trials": torch.randn(3, 133, 751), "labels": torch.zeros(3), "cache_version": CACHE_VERSION}, f3)

            lru.get(f1)
            lru.get(f2)
            self.assertEqual(len(lru), 2)

            lru.get(f3)
            self.assertEqual(len(lru), 2)
            self.assertNotIn(f1, lru._cache)
            self.assertIn(f2, lru._cache)
            self.assertIn(f3, lru._cache)

    @patch.object(EEGPreprocessingPipeline, "process_trials")
    def test_cache_hit_zero_pipeline_calls(self, mock_process_trials):
        """Test second initialization reads metadata index with 0 pipeline calls."""
        mock_process_trials.return_value = (
            np.random.randn(4, 4, 133, 751).astype(np.float32),
            np.zeros(4, dtype=np.int64),
            np.zeros(4, dtype=np.int64),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_cfg = {"enabled": True, "directory": tmp_dir}
            pipeline = EEGPreprocessingPipeline()

            ds1 = HGDDataset(file_paths=["/fake/sample.edf"], pipeline=pipeline, representation="frequency", cache_config=cache_cfg)
            self.assertEqual(mock_process_trials.call_count, 1)

            mock_process_trials.reset_mock()
            ds2 = HGDDataset(file_paths=["/fake/sample.edf"], pipeline=pipeline, representation="frequency", cache_config=cache_cfg)
            mock_process_trials.assert_not_called()
            self.assertEqual(len(ds1), len(ds2))


if __name__ == "__main__":
    unittest.main()
