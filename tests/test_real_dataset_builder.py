"""
Unit Tests for Real HGD DataLoader Integration & Factory (Phase 10 Patch v0.10.2).
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.builder import build_dataloaders, create_synthetic_dataset
from datasets.dataset import HGDDataset
from datasets.pipeline import EEGPreprocessingPipeline


class TestRealDatasetBuilder(unittest.TestCase):
    """Test suite for dataset builder, path discovery, train/val/test splits, and DataLoader factory."""

    def setUp(self):
        self.orig_env = os.environ.get("HGD_DATASET_ROOT")

    def tearDown(self):
        if self.orig_env is not None:
            os.environ["HGD_DATASET_ROOT"] = self.orig_env
        elif "HGD_DATASET_ROOT" in os.environ:
            del os.environ["HGD_DATASET_ROOT"]

    def test_synthetic_mode_flag(self):
        """Test synthetic_data: true constructs 3 valid synthetic DataLoaders."""
        config = {
            "training": {
                "synthetic_data": True,
                "batch_size": 16,
                "num_workers": 0,
                "seed": 42,
            },
            "model": {
                "num_bands": 4,
                "num_channels": 133,
                "num_samples": 250,
            },
        }

        train_loader, val_loader, test_loader = build_dataloaders(config, project_root=PROJECT_ROOT)

        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertIsNotNone(test_loader)

        x_b, y_b = next(iter(train_loader))
        self.assertEqual(x_b.shape, (16, 4, 133, 250))
        self.assertEqual(y_b.shape, (16,))

    def test_missing_dataset_raises_file_not_found(self):
        """Test missing dataset root raises FileNotFoundError when synthetic_data is False."""
        os.environ["HGD_DATASET_ROOT"] = os.path.join(PROJECT_ROOT, "non_existent_hgd_dir_9999")
        config = {
            "training": {
                "synthetic_data": False,
            }
        }

        with self.assertRaises(FileNotFoundError) as ctx:
            build_dataloaders(config, project_root=PROJECT_ROOT)

        self.assertIn("not found at", str(ctx.exception))

    @patch("datasets.builder.discover_edf_files")
    @patch.object(EEGPreprocessingPipeline, "process_trials")
    def test_real_dataset_builder_with_mocked_pipeline(self, mock_process_trials, mock_discover):
        """Test building DataLoaders with train/val split and test loader using mocked preprocessing."""
        # Setup mock returns
        mock_discover.side_effect = lambda path: (
            ["/fake/train1/1.edf", "/fake/train1/2.edf"]
            if "train1" in path
            else ["/fake/test1/1.edf"]
        )

        # Mock 2 trials per file (751 samples -> 11 windows per trial -> 22 windows per file)
        def fake_process_trials(filepath, representation="frequency"):
            x_trials = np.random.randn(2, 4, 133, 751).astype(np.float32)
            y_trials = np.random.randint(0, 4, size=(2,)).astype(np.int64)
            t_ids = np.arange(2, dtype=np.int64)
            return x_trials, y_trials, t_ids

        mock_process_trials.side_effect = fake_process_trials

        with tempfile.TemporaryDirectory() as tmp_dir:
            os.environ["HGD_DATASET_ROOT"] = tmp_dir
            os.makedirs(os.path.join(tmp_dir, "train1"), exist_ok=True)
            os.makedirs(os.path.join(tmp_dir, "test1"), exist_ok=True)

            config = {
                "training": {
                    "synthetic_data": False,
                    "batch_size": 4,
                    "num_workers": 0,
                    "validation_split": 0.2,
                    "split_strategy": "random",
                    "seed": 42,
                },
                "dataset": {
                    "representation": "frequency",
                },
                "model": {
                    "num_bands": 4,
                    "num_channels": 133,
                    "num_samples": 250,
                    "classifier": {"num_classes": 4},
                },
            }

            train_loader, val_loader, test_loader = build_dataloaders(config, project_root=PROJECT_ROOT)

            self.assertIsNotNone(train_loader)
            self.assertIsNotNone(val_loader)
            self.assertIsNotNone(test_loader)

            # 2 train files x 22 windows = 44 total train windows
            total_train_val = len(train_loader.dataset) + len(val_loader.dataset)
            self.assertEqual(total_train_val, 44)

            # Inspect train batch
            x_b, y_b = next(iter(train_loader))
            self.assertEqual(x_b.shape, (4, 4, 133, 250))
            self.assertEqual(y_b.shape, (4,))

    @patch("datasets.builder.discover_edf_files")
    @patch.object(EEGPreprocessingPipeline, "process_trials")
    def test_representation_configuration(self, mock_process_trials, mock_discover):
        """Test toggling representation configuration between frequency (4D) and time (3D)."""
        mock_discover.side_effect = lambda path: (
            ["/fake/train1/1.edf"] if "train1" in path else ["/fake/test1/1.edf"]
        )

        def fake_process_time(filepath, representation="time"):
            x_win = np.random.randn(2, 133, 751).astype(np.float32)
            y_win = np.random.randint(0, 4, size=(2,)).astype(np.int64)
            t_ids = np.zeros(2, dtype=np.int64)
            return x_win, y_win, t_ids

        mock_process_trials.side_effect = fake_process_time

        with tempfile.TemporaryDirectory() as tmp_dir:
            os.environ["HGD_DATASET_ROOT"] = tmp_dir
            os.makedirs(os.path.join(tmp_dir, "train1"), exist_ok=True)
            os.makedirs(os.path.join(tmp_dir, "test1"), exist_ok=True)

            config_time = {
                "training": {
                    "synthetic_data": False,
                    "batch_size": 2,
                    "num_workers": 0,
                    "validation_split": 0.2,
                },
                "dataset": {
                    "representation": "time",
                },
                "model": {
                    "classifier": {"num_classes": 4},
                },
            }

            train_loader, _, _ = build_dataloaders(config_time, project_root=PROJECT_ROOT)
            x_b, y_b = next(iter(train_loader))
            self.assertEqual(x_b.ndim, 3)  # (Batch, Channels, Samples)

    @patch.object(EEGPreprocessingPipeline, "process_trials")
    def test_dataset_cache_interface(self, mock_process_trials):
        """Test optional HGDDataset caching interface saves and reloads cached tensors."""
        mock_process_trials.return_value = (
            np.random.randn(2, 133, 751).astype(np.float32),
            np.zeros(2, dtype=np.int64),
            np.zeros(2, dtype=np.int64),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_cfg = {"enabled": True, "directory": tmp_dir}

            # First initialization (writes cache)
            ds1 = HGDDataset(
                file_paths=["/fake/sample.edf"],
                representation="time",
                cache_config=cache_cfg,
            )
            self.assertEqual(len(ds1), 22)

            # Check cache file created
            cache_files = os.listdir(tmp_dir)
            self.assertEqual(len(cache_files), 2)  # metadata.json + _trials.pt

            # Second initialization (loads from cache without calling process_trials again)
            mock_process_trials.reset_mock()
            ds2 = HGDDataset(
                file_paths=["/fake/sample.edf"],
                representation="time",
                cache_config=cache_cfg,
            )
            mock_process_trials.assert_not_called()
            self.assertEqual(len(ds1), len(ds2))
            torch.testing.assert_close(ds1.X, ds2.X)


if __name__ == "__main__":
    unittest.main()
