"""
Unit Tests for Augmentation Strategies (Phase 10).
"""

import os
import sys
import unittest
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from augmentation.strategies import WGANGPStrategy, MixUpStrategy, CutMixStrategy, NoAugmentationStrategy
from augmentation.gan.generator import ConditionalEEGGenerator


class TestAugmentationStrategies(unittest.TestCase):
    """Test suite for augmentation strategy implementations."""

    def test_mixup_strategy(self):
        """Test MixUpStrategy augments real dataset by ratio."""
        strat = MixUpStrategy(seed=42)
        real_x = torch.randn(10, 2, 5, 20)
        real_y = torch.randint(0, 4, (10,))

        aug_x, aug_y = strat.augment(real_x, real_y, ratio=0.5)
        self.assertEqual(aug_x.shape[0], 15)
        self.assertEqual(aug_y.shape[0], 15)

    def test_wgan_gp_strategy_generate(self):
        """Test WGANGPStrategy generate produces class-balanced SyntheticDataset."""
        strat = WGANGPStrategy(seed=42)
        strat.generator = ConditionalEEGGenerator(latent_dim=16, num_classes=4, num_bands=2, num_channels=5, num_samples=20)
        strat.device = torch.device("cpu")

        syn_ds = strat.generate(num_samples=12, num_classes=4)
        syn_x = syn_ds.get_data()
        syn_y = syn_ds.get_labels()

        self.assertEqual(syn_x.shape, (12, 2, 5, 20))
        self.assertEqual(syn_y.shape, (12,))


if __name__ == "__main__":
    unittest.main()
