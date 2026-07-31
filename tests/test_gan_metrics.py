"""
Unit Tests for Standardized GAN Metrics (Phase 10).
"""

import os
import sys
import unittest
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from augmentation.metrics import PSDSimilarity, BandPowerSimilarity, CovarianceDistance, DiversityScore


class TestGANMetrics(unittest.TestCase):
    """Test suite for GAN quality evaluation metrics."""

    def test_gan_metrics_outputs(self):
        """Test GAN metrics compute finite non-NaN numerical values."""
        real_eeg = torch.randn(8, 2, 5, 20)
        fake_eeg = torch.randn(8, 2, 5, 20)

        psd_sim = PSDSimilarity().compute(real_eeg, fake_eeg)
        bp_sim = BandPowerSimilarity().compute(real_eeg, fake_eeg)
        cov_dist = CovarianceDistance().compute(real_eeg, fake_eeg)
        div_score = DiversityScore().compute(real_eeg, fake_eeg)

        self.assertTrue(0.0 <= psd_sim <= 1.0)
        self.assertTrue(0.0 <= bp_sim <= 1.0)
        self.assertTrue(cov_dist >= 0.0)
        self.assertTrue(div_score >= 0.0)


if __name__ == "__main__":
    unittest.main()
