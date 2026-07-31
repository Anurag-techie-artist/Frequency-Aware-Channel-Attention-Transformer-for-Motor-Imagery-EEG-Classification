"""
Unit Tests for Seed Management (Phase 7).
"""

import os
import sys
import unittest
import torch
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.seed import set_seed


class TestSeed(unittest.TestCase):
    """Test suite for reproducibility seed utility."""

    def test_reproducibility(self):
        """Test set_seed ensures reproducible random numbers across runs."""
        set_seed(42)
        r1_py = np.random.rand(5)
        r1_torch = torch.randn(5)

        set_seed(42)
        r2_py = np.random.rand(5)
        r2_torch = torch.randn(5)

        self.assertTrue(np.allclose(r1_py, r2_py))
        self.assertTrue(torch.equal(r1_torch, r2_torch))


if __name__ == "__main__":
    unittest.main()
