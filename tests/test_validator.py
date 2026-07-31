"""
Unit Tests for Synthetic Data Validator (Phase 10).
"""

import os
import sys
import unittest
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from augmentation.validator import SyntheticDataValidator


class TestSyntheticDataValidator(unittest.TestCase):
    """Test suite for SyntheticDataValidator integrity checks."""

    def test_valid_dataset(self):
        """Test validator passes valid tensor shapes and values."""
        x = torch.randn(10, 4, 133, 250)
        y = torch.randint(0, 4, (10,))
        self.assertTrue(SyntheticDataValidator.validate_synthetic_dataset(x, y))

    def test_nan_detection(self):
        """Test validator raises ValueError when tensor contains NaN."""
        x = torch.randn(10, 4, 133, 250)
        x[0, 0, 0, 0] = float("nan")
        y = torch.randint(0, 4, (10,))

        with self.assertRaises(ValueError):
            SyntheticDataValidator.validate_synthetic_dataset(x, y)


if __name__ == "__main__":
    unittest.main()
