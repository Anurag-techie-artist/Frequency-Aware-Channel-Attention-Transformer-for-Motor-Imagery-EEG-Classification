"""
Unit Tests for Device Resolution (Phase 7).
"""

import os
import sys
import unittest
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.device import get_device


class TestDevice(unittest.TestCase):
    """Test suite for get_device resolution utility."""

    def test_device_resolution(self):
        """Test CPU and auto device resolution."""
        dev_cpu = get_device("cpu")
        self.assertEqual(dev_cpu.type, "cpu")

        dev_auto = get_device("auto")
        self.assertIn(dev_auto.type, ["cpu", "cuda", "mps"])


if __name__ == "__main__":
    unittest.main()
