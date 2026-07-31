"""
Unit Tests for Conditional EEG Generator (Phase 10).
"""

import os
import sys
import unittest
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from augmentation.gan.generator import ConditionalEEGGenerator


class TestConditionalEEGGenerator(unittest.TestCase):
    """Test suite for ConditionalEEGGenerator tensor shape outputs."""

    def test_generator_forward_shape(self):
        """Test generator maps noise and labels to (B, Bands, Channels, Samples) shape."""
        net = ConditionalEEGGenerator(
            latent_dim=64, num_classes=4, num_bands=2, num_channels=10, num_samples=50
        )
        noise = torch.randn(8, 64)
        labels = torch.randint(0, 4, (8,))

        fake_eeg = net(noise, labels)
        self.assertEqual(fake_eeg.shape, (8, 2, 10, 50))


if __name__ == "__main__":
    unittest.main()
