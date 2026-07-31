"""
Unit Tests for CLSToken Layer (Phase 5.1).

Tests:
1. test_shape_expansion: Verify (B, N, d_model) -> (B, N+1, d_model).
2. test_cls_token_indexing: Verify token at index 0 matches expanded CLS parameter.
3. test_gradient_flow: Verify gradients propagate to self.cls_token parameter.
4. test_serialization: Verify state_dict serialization.
"""

import os
import sys
import tempfile
import unittest
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.common import CLSToken


class TestCLSToken(unittest.TestCase):
    """Test suite for CLSToken module."""

    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 4
        self.num_tokens = 532
        self.d_model = 128
        self.sample_x = torch.randn(self.batch_size, self.num_tokens, self.d_model)

    def test_shape_expansion(self):
        """Test output shape (B, N+1, d_model)."""
        cls_layer = CLSToken(d_model=self.d_model)
        out_4d = cls_layer(self.sample_x)
        self.assertEqual(out_4d.shape, (self.batch_size, self.num_tokens + 1, self.d_model))

        sample_2d = self.sample_x[0]
        out_2d = cls_layer(sample_2d)
        self.assertEqual(out_2d.shape, (self.num_tokens + 1, self.d_model))

    def test_cls_token_indexing(self):
        """Test token at index 0 is CLS token and indices 1..N match input."""
        cls_layer = CLSToken(d_model=self.d_model)
        out = cls_layer(self.sample_x)

        # Token 0 should match cls_token expanded
        expected_cls = cls_layer.cls_token.expand(self.batch_size, -1, -1)
        self.assertTrue(torch.equal(out[:, 0:1, :], expected_cls))

        # Remaining tokens should match input
        self.assertTrue(torch.equal(out[:, 1:, :], self.sample_x))

    def test_gradient_flow(self):
        """Test gradients flow to self.cls_token."""
        cls_layer = CLSToken(d_model=self.d_model)
        out = cls_layer(self.sample_x)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(cls_layer.cls_token.grad)
        self.assertFalse(torch.isnan(cls_layer.cls_token.grad).any())
        self.assertFalse(torch.all(cls_layer.cls_token.grad == 0))

    def test_serialization(self):
        """Test serialization parity."""
        cls_layer = CLSToken(d_model=self.d_model)
        out_orig = cls_layer(self.sample_x)

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt = os.path.join(tmp_dir, "cls.pt")
            torch.save(cls_layer.state_dict(), ckpt)

            loaded = CLSToken(d_model=self.d_model)
            loaded.load_state_dict(torch.load(ckpt))

            out_loaded = loaded(self.sample_x)
            self.assertTrue(torch.equal(out_orig, out_loaded))


if __name__ == "__main__":
    unittest.main()
