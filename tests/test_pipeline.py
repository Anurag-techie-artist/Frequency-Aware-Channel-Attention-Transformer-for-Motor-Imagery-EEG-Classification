"""
Unit Tests for AugmentationPipeline (Phase 10).
"""

import os
import sys
import unittest
import torch
from torch.utils.data import TensorDataset, DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from augmentation.strategies import MixUpStrategy
from augmentation.pipeline import AugmentationPipeline
from datasets.pipeline import PreprocessingConfig, EEGPreprocessingPipeline
from configs.config_loader import load_master_config


class TestAugmentationPipeline(unittest.TestCase):
    """Test suite for AugmentationPipeline DataLoader construction."""

    def test_pipeline_dataloader_augmentation(self):
        """Test AugmentationPipeline creates augmented DataLoader with combined size."""
        dataset = TensorDataset(torch.randn(10, 2, 5, 20), torch.randint(0, 4, (10,)))
        loader = DataLoader(dataset, batch_size=4)

        strat = MixUpStrategy(seed=42)
        pipeline = AugmentationPipeline(strat)
        aug_loader = pipeline.augment_dataloader(loader, ratio=0.5, batch_size=4)

        total_samples = sum(b[0].shape[0] for b in aug_loader)
        self.assertEqual(total_samples, 15)


class TestPreprocessingConfigTypeNormalization(unittest.TestCase):
    """Regression test suite for v0.10.3 Configuration Type Normalization."""

    def test_master_config_pipeline_direct_construction(self):
        """Test passing master configuration dictionary directly to EEGPreprocessingPipeline (regression test)."""
        master_config = load_master_config(project_root=PROJECT_ROOT)
        pipeline = EEGPreprocessingPipeline(config=master_config)
        config = pipeline.config

        self.assertIsInstance(config.eps, float)
        self.assertIsInstance(config.window_size, int)
        self.assertIsInstance(config.window_stride, int)
        self.assertIsInstance(config.sampling_rate, float)
        self.assertIsInstance(config.filter_low, float)
        self.assertIsInstance(config.filter_high, float)
        self.assertIsInstance(config.epoch_start, float)
        self.assertIsInstance(config.epoch_end, float)
        self.assertIsInstance(config.normalization, str)

    def test_stringified_and_mixed_type_normalization(self):
        """Test PreprocessingConfig normalizes stringified floats and ints on instantiation."""
        raw_dict = {
            "sampling_rate": "250.0",
            "filter_low": "4.0",
            "filter_high": "38",
            "epoch_start": "0.5",
            "epoch_end": "3.5",
            "window_size": "250",
            "window_stride": "50",
            "normalization": "zscore",
            "eps": "1e-6",
        }
        config = PreprocessingConfig.from_dict(raw_dict)

        self.assertIsInstance(config.eps, float)
        self.assertEqual(config.eps, 1e-6)
        self.assertIsInstance(config.window_size, int)
        self.assertEqual(config.window_size, 250)
        self.assertIsInstance(config.window_stride, int)
        self.assertEqual(config.window_stride, 50)
        self.assertIsInstance(config.sampling_rate, float)
        self.assertEqual(config.sampling_rate, 250.0)

    def test_direct_dataclass_instantiation_type_normalization(self):
        """Test PreprocessingConfig __post_init__ normalizes types even on direct instantiation."""
        config = PreprocessingConfig(
            window_size="250",
            window_stride="50",
            eps="1e-6",
        )
        self.assertIsInstance(config.eps, float)
        self.assertIsInstance(config.window_size, int)
        self.assertIsInstance(config.window_stride, int)

    def test_from_dict_filters_unrelated_sections(self):
        """Test PreprocessingConfig.from_dict filters out model/training sections cleanly."""
        master_dict = {
            "training": {"batch_size": 32},
            "model": {"num_bands": 4},
            "window_size": 250,
            "eps": 1e-6,
        }
        config = PreprocessingConfig.from_dict(master_dict)
        self.assertEqual(config.window_size, 250)
        self.assertEqual(config.eps, 1e-6)
        self.assertIsNotNone(config.raw_dict)
        self.assertEqual(config.raw_dict, master_dict)

    def test_invalid_config_values_validation(self):
        """Test PreprocessingConfig validates value boundaries and raises ValueError."""
        with self.assertRaises(ValueError):
            PreprocessingConfig(window_size=-10)
        with self.assertRaises(ValueError):
            PreprocessingConfig(eps=0)
        with self.assertRaises(ValueError):
            PreprocessingConfig(sampling_rate=-250)


if __name__ == "__main__":
    unittest.main()

