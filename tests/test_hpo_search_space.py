"""
Unit Tests for HPO Search Space and Parameter Sampling (Phase 9).
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from hpo.search_space import SearchSpace


class TestHPOSearchSpace(unittest.TestCase):
    """Test suite for search space parameter sampling and range bounds."""

    def test_search_space_sampling(self):
        """Test parameter sampling produces values within defined bounds."""
        config = {
            "learning_rate": {"type": "loguniform", "low": 1e-4, "high": 1e-2},
            "weight_decay": {"type": "float", "distribution": "uniform", "low": 0.001, "high": 0.01},
            "batch_size": {"type": "categorical", "values": [16, 32, 64]},
            "num_layers": {"type": "integer", "low": 1, "high": 4, "step": 1},
        }

        ss = SearchSpace(config)
        self.assertEqual(len(ss), 4)

        sample = ss.sample()
        self.assertTrue(1e-4 <= sample["learning_rate"] <= 1e-2)
        self.assertTrue(0.001 <= sample["weight_decay"] <= 0.01)
        self.assertIn(sample["batch_size"], [16, 32, 64])
        self.assertTrue(1 <= sample["num_layers"] <= 4)


if __name__ == "__main__":
    unittest.main()
