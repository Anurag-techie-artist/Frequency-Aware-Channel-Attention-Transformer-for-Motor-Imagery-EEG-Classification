"""
Unit Tests for EEGMotorImageryModel Assembly (Phase 7).

Tests:
1. test_forward_pass: (B, F, C, S) -> (B, num_classes).
2. test_from_config: Model instantiation directly from config dict.
3. test_prediction_output: return_metadata=True returns PredictionOutput dataclass.
"""

import os
import sys
import unittest
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.eeg_motor_imagery_model import EEGMotorImageryModel
from models.classifier import PredictionOutput
from configs.config_loader import load_master_config


class TestEEGMotorImageryModel(unittest.TestCase):
    """Test suite for assembled EEGMotorImageryModel module."""

    def setUp(self):
        torch.manual_seed(42)
        self.config = load_master_config()
        self.sample_x = torch.randn(2, 4, 133, 250)

    def test_from_config(self):
        """Test instantiation from config dict."""
        model = EEGMotorImageryModel.from_config(self.config)
        model.eval()

        logits = model(self.sample_x)
        self.assertEqual(logits.shape, (2, 4))

    def test_prediction_output(self):
        """Test return_metadata=True returns PredictionOutput."""
        model = EEGMotorImageryModel.from_config(self.config)
        model.eval()

        pred = model(self.sample_x, return_metadata=True)
        self.assertIsInstance(pred, PredictionOutput)
        self.assertEqual(pred.logits.shape, (2, 4))
        self.assertEqual(pred.probabilities.shape, (2, 4))
        self.assertEqual(pred.predicted_class.shape, (2,))


if __name__ == "__main__":
    unittest.main()
