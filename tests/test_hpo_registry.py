"""
Unit Tests for PARAMETER_REGISTRY and STRATEGY_REGISTRY (Phase 9).
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from hpo.registry import PARAMETER_REGISTRY, STRATEGY_REGISTRY, register_parameter, register_strategy
from hpo.parameters.base import Parameter
from hpo.strategies.base import SearchStrategy


class CustomParam(Parameter):
    def sample(self, rng=None):
        return 42
    def validate(self):
        return True


class CustomStrategy(SearchStrategy):
    def suggest(self, search_space, trial_id):
        return {"custom": 42}


class TestHPORegistry(unittest.TestCase):
    """Test suite for registry lookup and dynamic extension."""

    def test_parameter_registry(self):
        """Test PARAMETER_REGISTRY contains standard types and allows custom registration."""
        self.assertIn("float", PARAMETER_REGISTRY)
        self.assertIn("categorical", PARAMETER_REGISTRY)

        register_parameter("custom", CustomParam)
        self.assertIn("custom", PARAMETER_REGISTRY)

    def test_strategy_registry(self):
        """Test STRATEGY_REGISTRY contains standard strategies and allows custom registration."""
        self.assertIn("random", STRATEGY_REGISTRY)
        self.assertIn("grid", STRATEGY_REGISTRY)

        register_strategy("custom_strat", CustomStrategy)
        self.assertIn("custom_strat", STRATEGY_REGISTRY)


if __name__ == "__main__":
    unittest.main()
