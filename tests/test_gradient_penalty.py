"""
Unit Tests for Gradient Penalty Computation (Phase 10).
"""

import os
import sys
import unittest
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from augmentation.gan.critic import ConditionalEEGCritic
from augmentation.gan.gradient_penalty import compute_gradient_penalty


class TestGradientPenalty(unittest.TestCase):
    """Test suite for WGAN-GP 1-Lipschitz gradient penalty computation."""

    def test_gradient_penalty_computation(self):
        """Test gradient penalty returns a scalar non-negative loss tensor."""
        critic = ConditionalEEGCritic(num_classes=4, num_bands=2, num_channels=10, num_samples=50)
        real_eeg = torch.randn(4, 2, 10, 50)
        fake_eeg = torch.randn(4, 2, 10, 50)
        labels = torch.randint(0, 4, (4,))

        gp = compute_gradient_penalty(critic, real_eeg, fake_eeg, labels, torch.device("cpu"))
        self.assertEqual(gp.ndim, 0)
        self.assertTrue(gp.item() >= 0.0)


if __name__ == "__main__":
    unittest.main()
