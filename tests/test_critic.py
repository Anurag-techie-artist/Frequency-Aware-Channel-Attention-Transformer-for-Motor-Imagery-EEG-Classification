"""
Unit Tests for Conditional EEG Critic (Phase 10).
"""

import os
import sys
import unittest
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from augmentation.gan.critic import ConditionalEEGCritic


class TestConditionalEEGCritic(unittest.TestCase):
    """Test suite for ConditionalEEGCritic scalar score outputs."""

    def test_critic_forward_shape(self):
        """Test critic maps input EEG tensor and labels to (B, 1) scalar scores."""
        critic = ConditionalEEGCritic(
            num_classes=4, num_bands=2, num_channels=10, num_samples=50
        )
        eeg = torch.randn(8, 2, 10, 50)
        labels = torch.randint(0, 4, (8,))

        score = critic(eeg, labels)
        self.assertEqual(score.shape, (8, 1))


if __name__ == "__main__":
    unittest.main()
