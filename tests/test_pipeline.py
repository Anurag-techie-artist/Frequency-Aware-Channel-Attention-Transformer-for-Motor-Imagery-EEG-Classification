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


if __name__ == "__main__":
    unittest.main()
