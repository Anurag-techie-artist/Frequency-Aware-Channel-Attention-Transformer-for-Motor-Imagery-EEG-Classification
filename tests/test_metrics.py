"""
Unit Tests for Metrics Package (Phase 8).
"""

import os
import sys
import unittest
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from metrics import (
    compute_accuracy,
    compute_balanced_accuracy,
    compute_cohen_kappa,
    compute_confusion_matrix,
    compute_classification_metrics,
)


class TestMetrics(unittest.TestCase):
    """Test suite for evaluation metrics functions."""

    def setUp(self):
        torch.manual_seed(42)
        self.targets = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])

        # Perfect predictions logits
        self.perfect_logits = torch.zeros(8, 4)
        for i, t in enumerate(self.targets):
            self.perfect_logits[i, t] = 10.0

        # Mixed predictions logits
        self.mixed_preds = torch.tensor([0, 1, 2, 0, 0, 1, 3, 3])
        self.mixed_logits = torch.zeros(8, 4)
        for i, p in enumerate(self.mixed_preds):
            self.mixed_logits[i, p] = 10.0

    def test_perfect_metrics(self):
        """Test metrics compute 1.0 for perfect predictions."""
        acc = compute_accuracy(self.perfect_logits, self.targets)
        bal_acc = compute_balanced_accuracy(self.perfect_logits, self.targets, num_classes=4)
        kappa = compute_cohen_kappa(self.perfect_logits, self.targets, num_classes=4)

        self.assertEqual(acc, 1.0)
        self.assertEqual(bal_acc, 1.0)
        self.assertEqual(kappa, 1.0)

    def test_confusion_matrix_shape(self):
        """Test confusion matrix shape is K x K."""
        cm = compute_confusion_matrix(self.mixed_logits, self.targets, num_classes=4)
        self.assertEqual(cm.shape, (4, 4))
        self.assertEqual(cm.sum().item(), 8)

    def test_comprehensive_classification_metrics(self):
        """Test compute_classification_metrics returns expected dictionary structure."""
        metrics = compute_classification_metrics(self.mixed_logits, self.targets, num_classes=4)

        self.assertIn("accuracy", metrics)
        self.assertIn("balanced_accuracy", metrics)
        self.assertIn("cohen_kappa", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1", metrics)
        self.assertIn("weighted_f1", metrics)
        self.assertIn("confusion_matrix", metrics)
        self.assertIn("per_class", metrics)


if __name__ == "__main__":
    unittest.main()
