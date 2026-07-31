"""
Unit Tests for Evaluator and Report Generation (Phase 8).
"""

import os
import sys
import tempfile
import unittest
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.eeg_motor_imagery_model import EEGMotorImageryModel
from datasets.builder import create_synthetic_dataset
from evaluation.evaluator import Evaluator


class TestEvaluator(unittest.TestCase):
    """Test suite for Evaluator and ReportGenerator."""

    def test_evaluator_execution_and_reports(self):
        """Test evaluator generates manifest, metrics, JSON, CSV, and raw predictions."""
        config = {
            "model": {
                "transformer": {"d_model": 32, "num_layers": 1},
                "classifier": {"hidden_dim": 64},
            }
        }
        model = EEGMotorImageryModel.from_config(config)
        ds = create_synthetic_dataset(num_samples=16)
        loader = DataLoader(ds, batch_size=8)

        with tempfile.TemporaryDirectory() as tmp_dir:
            evaluator = Evaluator(model=model, output_dir=tmp_dir)
            metrics, results = evaluator.evaluate(
                dataloader=loader,
                config=config,
                checkpoint_name="test.pt",
                generate_plots=False,
            )

            self.assertIn("accuracy", metrics)
            self.assertEqual(results.logits.shape, (16, 4))
            self.assertEqual(results.cls_embeddings.shape, (16, 32))

            # Verify exported files exist
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "manifest.json")))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "evaluation_report.json")))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "metrics.csv")))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "predictions.csv")))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "raw_predictions.pt")))


if __name__ == "__main__":
    unittest.main()
