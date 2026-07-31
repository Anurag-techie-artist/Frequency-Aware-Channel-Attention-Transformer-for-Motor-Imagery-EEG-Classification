"""
Unit Tests for Report Reproducibility (Phase 8).

Verifies deterministic evaluation output across repeated evaluation runs.
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


class TestReportReproducibility(unittest.TestCase):
    """Test suite for deterministic evaluation output reproducibility."""

    def test_reproducible_evaluation_outputs(self):
        """Test repeated evaluations yield identical metrics and prediction outputs."""
        config = {
            "model": {
                "transformer": {"d_model": 32, "num_layers": 1},
                "classifier": {"hidden_dim": 64},
            }
        }
        model = EEGMotorImageryModel.from_config(config)
        ds = create_synthetic_dataset(num_samples=16, seed=42)
        loader = DataLoader(ds, batch_size=8)

        with tempfile.TemporaryDirectory() as tmp_dir1, tempfile.TemporaryDirectory() as tmp_dir2:
            evaluator1 = Evaluator(model=model, output_dir=tmp_dir1)
            metrics1, res1 = evaluator1.evaluate(dataloader=loader, config=config, generate_plots=False)

            evaluator2 = Evaluator(model=model, output_dir=tmp_dir2)
            metrics2, res2 = evaluator2.evaluate(dataloader=loader, config=config, generate_plots=False)

            self.assertEqual(metrics1["accuracy"], metrics2["accuracy"])
            self.assertEqual(metrics1["f1"], metrics2["f1"])
            self.assertEqual(metrics1["cohen_kappa"], metrics2["cohen_kappa"])
            self.assertTrue(torch.equal(res1.logits, res2.logits))
            self.assertTrue(torch.equal(res1.cls_embeddings, res2.cls_embeddings))


if __name__ == "__main__":
    unittest.main()
