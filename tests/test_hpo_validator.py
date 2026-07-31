"""
Unit Tests for Search Space Validator (Phase 9).
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from hpo.validator import SearchSpaceValidator


class TestHPOValidator(unittest.TestCase):
    """Test suite for catching search space configuration errors upfront."""

    def test_invalid_ranges(self):
        """Test validator raises error when low >= high."""
        cfg = {"lr": {"type": "float", "low": 0.5, "high": 0.1}}
        with self.assertRaises(ValueError):
            SearchSpaceValidator.validate_search_space_config(cfg)

    def test_empty_categorical(self):
        """Test validator raises error on empty categorical values."""
        cfg = {"batch_size": {"type": "categorical", "values": []}}
        with self.assertRaises(ValueError):
            SearchSpaceValidator.validate_search_space_config(cfg)

    def test_loguniform_negative(self):
        """Test validator raises error when loguniform low <= 0."""
        cfg = {"lr": {"type": "loguniform", "low": -0.01, "high": 0.1}}
        with self.assertRaises(ValueError):
            SearchSpaceValidator.validate_search_space_config(cfg)


if __name__ == "__main__":
    unittest.main()
