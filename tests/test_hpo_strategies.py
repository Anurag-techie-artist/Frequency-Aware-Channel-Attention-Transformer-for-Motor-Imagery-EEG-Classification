"""
Unit Tests for Search Strategies (Phase 9).
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from hpo.search_space import SearchSpace
from hpo.strategies import RandomSearchStrategy, GridSearchStrategy, OptunaSearchStrategy


class TestHPOStrategies(unittest.TestCase):
    """Test suite for parameter suggestions from search strategies."""

    def setUp(self):
        self.config = {
            "batch_size": {"type": "categorical", "values": [16, 32]},
            "d_model": {"type": "categorical", "values": [64, 128]},
        }
        self.ss = SearchSpace(self.config)

    def test_random_search_suggest(self):
        """Test RandomSearchStrategy generates parameters."""
        strat = RandomSearchStrategy(seed=42)
        params = strat.suggest(self.ss, trial_id=0)
        self.assertIn("batch_size", params)
        self.assertIn("d_model", params)

    def test_grid_search_suggest(self):
        """Test GridSearchStrategy enumerates grid combinations."""
        strat = GridSearchStrategy(seed=42)
        p0 = strat.suggest(self.ss, trial_id=0)
        p1 = strat.suggest(self.ss, trial_id=1)
        self.assertNotEqual(p0, p1)

    def test_optuna_search_fallback(self):
        """Test OptunaSearchStrategy initializes and suggests parameters."""
        strat = OptunaSearchStrategy(seed=42)
        params = strat.suggest(self.ss, trial_id=0)
        self.assertIn("batch_size", params)


if __name__ == "__main__":
    unittest.main()
