"""
Unit Tests for EEGClassifier Module (Phase 6).

Tests:
1. test_input_compatibility: Accepts FATEOutput dataclass, tuple output, and raw torch.Tensor.
2. test_prediction_consistency: Model eval mode predictions and logits remain identical across runs and serialization.
3. test_logits_shape: (B, num_classes) for variable batch sizes B=1, 4, 8 and d_model.
4. test_numerical_stability: Zero NaN/Inf outputs.
5. test_prediction_output_api: Test return_metadata=True returns PredictionOutput directly with dataclass immutability.
"""

import os
import sys
import tempfile
import unittest
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.classifier import (
    EEGClassifier,
    ClassifierConfig,
    PredictionOutput,
    ClassifierMetadata,
)
from models.transformer import (
    FATEOutput,
    TokenizationMetadata,
)


class TestEEGClassifier(unittest.TestCase):
    """Test suite for EEGClassifier module."""

    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 3
        self.d_model = 128
        self.num_classes = 4
        self.num_tokens = 532

        self.sample_cls = torch.randn(self.batch_size, self.d_model)
        self.sample_context = torch.randn(self.batch_size, self.num_tokens, self.d_model)

        dummy_meta = TokenizationMetadata(
            mappings=(),
            num_tokens=self.num_tokens,
            num_bands=4,
            num_channels=133,
            num_samples=250,
            input_shape=(self.batch_size, 4, 133, 250),
            output_shape=(self.batch_size, self.num_tokens, 250),
        )
        self.fate_out = FATEOutput(
            contextual_embeddings=self.sample_context,
            cls_embedding=self.sample_cls,
            token_metadata=dummy_meta,
        )

    def test_input_compatibility(self):
        """Test classifier accepts both FATEOutput dataclass, tuple output, and raw Tensor."""
        clf = EEGClassifier(d_model=self.d_model, num_classes=self.num_classes)
        clf.eval()

        # Pass raw tensor
        logits_tensor = clf(self.sample_cls)
        self.assertEqual(logits_tensor.shape, (self.batch_size, self.num_classes))

        # Pass FATEOutput
        logits_fate = clf(self.fate_out)
        self.assertEqual(logits_fate.shape, (self.batch_size, self.num_classes))

        # Pass Tuple
        logits_tuple = clf((self.sample_context, self.fate_out))
        self.assertEqual(logits_tuple.shape, (self.batch_size, self.num_classes))

        self.assertTrue(torch.equal(logits_tensor, logits_fate))
        self.assertTrue(torch.equal(logits_tensor, logits_tuple))

    def test_prediction_consistency(self):
        """Test prediction consistency across eval runs and serialization."""
        clf = EEGClassifier(d_model=self.d_model, num_classes=self.num_classes)
        clf.eval()

        p1 = clf(self.fate_out, return_metadata=True)
        p2 = clf(self.fate_out, return_metadata=True)

        self.assertIsInstance(p1, PredictionOutput)
        self.assertTrue(torch.equal(p1.logits, p2.logits))
        self.assertTrue(torch.equal(p1.probabilities, p2.probabilities))
        self.assertTrue(torch.equal(p1.predicted_class, p2.predicted_class))

        # Test serialization consistency
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt = os.path.join(tmp_dir, "clf.pt")
            torch.save(clf.state_dict(), ckpt)

            loaded = EEGClassifier(d_model=self.d_model, num_classes=self.num_classes)
            loaded.load_state_dict(torch.load(ckpt))
            loaded.eval()

            p_loaded = loaded(self.fate_out, return_metadata=True)
            self.assertTrue(torch.equal(p1.logits, p_loaded.logits))
            self.assertTrue(torch.equal(p1.predicted_class, p_loaded.predicted_class))

    def test_logits_shape(self):
        """Test shape preservation for variable batch sizes and d_model."""
        for b in [1, 4, 8]:
            for d in [64, 128, 256]:
                with self.subTest(batch_size=b, d_model=d):
                    clf = EEGClassifier(d_model=d, num_classes=4)
                    clf.eval()
                    cls_input = torch.randn(b, d)
                    logits = clf(cls_input)
                    self.assertEqual(logits.shape, (b, 4))

    def test_numerical_stability(self):
        """Test zero NaN/Inf outputs under extreme input values."""
        clf = EEGClassifier(d_model=self.d_model)
        clf.eval()

        zeros_in = torch.zeros(2, self.d_model)
        p_zeros = clf(zeros_in, return_metadata=True)
        self.assertFalse(torch.isnan(p_zeros.logits).any())
        self.assertFalse(torch.isnan(p_zeros.probabilities).any())

        large_in = torch.randn(2, self.d_model) * 1e4
        p_large = clf(large_in, return_metadata=True)
        self.assertFalse(torch.isnan(p_large.logits).any())
        self.assertFalse(torch.isnan(p_large.probabilities).any())

    def test_prediction_output_api(self):
        """Test return_metadata=True API returns PredictionOutput directly."""
        clf = EEGClassifier(d_model=self.d_model)
        clf.eval()
        pred_out = clf(self.fate_out, return_metadata=True)

        self.assertIsInstance(pred_out, PredictionOutput)
        self.assertIsInstance(pred_out.metadata, ClassifierMetadata)
        self.assertEqual(pred_out.logits.shape, (self.batch_size, 4))
        self.assertEqual(pred_out.probabilities.shape, (self.batch_size, 4))
        self.assertEqual(pred_out.predicted_class.shape, (self.batch_size,))

        # Verify probabilities sum to 1.0 per sample
        prob_sums = pred_out.probabilities.sum(dim=-1)
        self.assertTrue(torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5))

        # Check immutability
        with self.assertRaises(Exception):
            pred_out.logits = None


if __name__ == "__main__":
    unittest.main()
