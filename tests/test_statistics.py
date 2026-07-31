"""
Unit Tests for Statistical Validation Package (Phase 10).
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from augmentation.statistics import (
    compute_confidence_interval,
    compute_statistical_significance,
    compute_effect_size,
    bootstrap_resample,
)


class TestAugmentationStatistics(unittest.TestCase):
    """Test suite for statistical confidence intervals, p-values, and effect size functions."""

    def test_confidence_interval(self):
        """Test compute_confidence_interval outputs mean and CI bounds."""
        vals = [0.85, 0.87, 0.88, 0.86, 0.89]
        mean, lower, upper = compute_confidence_interval(vals)
        self.assertAlmostEqual(mean, 0.87, places=2)
        self.assertTrue(lower <= mean <= upper)

    def test_significance_and_effect_size(self):
        """Test compute_statistical_significance and compute_effect_size."""
        baseline = [0.70, 0.72, 0.71, 0.69, 0.70]
        augmented = [0.85, 0.86, 0.84, 0.87, 0.85]

        sig = compute_statistical_significance(baseline, augmented)
        eff = compute_effect_size(baseline, augmented)

        self.assertTrue(sig["is_statistically_significant"])
        self.assertTrue(eff["cohens_d"] > 1.0)


if __name__ == "__main__":
    unittest.main()
