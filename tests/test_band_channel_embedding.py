"""
Unit Tests for BandChannelEmbedding (Phase 5).

Tests:
1. test_projection_dimensions: Verify (B, N, S) -> (B, N, d_model).
2. test_embedding_addition: Verify summation of sample projection + Band + Channel embeddings.
3. test_deterministic_inference: Verify model.eval() produces identical outputs.
4. test_serialization: Verify torch.save() & torch.load() parity.
5. test_gradient_flow: Verify loss backward computes valid non-zero gradients.
"""

import os
import sys
import tempfile
import unittest
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.transformer import BandChannelEmbedding, EmbeddingConfig


class TestBandChannelEmbedding(unittest.TestCase):
    """Test suite for BandChannelEmbedding PyTorch module."""

    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 2
        self.num_bands = 4
        self.num_channels = 133
        self.num_samples = 250
        self.num_tokens = self.num_bands * self.num_channels
        self.d_model = 128

        self.sample_tokens = torch.randn(
            self.batch_size, self.num_tokens, self.num_samples
        )

    def test_projection_dimensions(self):
        """Test output shape (B, N, d_model)."""
        embedder = BandChannelEmbedding(d_model=self.d_model)

        out_4d = embedder(
            self.sample_tokens,
            num_bands=self.num_bands,
            num_channels=self.num_channels,
        )
        self.assertEqual(
            out_4d.shape, (self.batch_size, self.num_tokens, self.d_model)
        )

        out_3d = embedder(
            self.sample_tokens[0],
            num_bands=self.num_bands,
            num_channels=self.num_channels,
        )
        self.assertEqual(out_3d.shape, (self.num_tokens, self.d_model))

    def test_embedding_addition(self):
        """Test that output contains projection + band embedding + channel embedding."""
        embedder = BandChannelEmbedding(d_model=self.d_model, dropout=0.0)
        embedder.eval()

        out = embedder(
            self.sample_tokens,
            num_bands=self.num_bands,
            num_channels=self.num_channels,
        )

        # Calculate manual sum
        proj = embedder.temporal_projection(self.sample_tokens)
        b_idx = torch.arange(self.num_tokens) // self.num_channels
        c_idx = torch.arange(self.num_tokens) % self.num_channels

        b_emb = embedder.band_embedding(b_idx)
        c_emb = embedder.channel_embedding(c_idx)
        expected = proj + (b_emb + c_emb).unsqueeze(0)

        self.assertTrue(torch.allclose(out, expected, atol=1e-5))

    def test_deterministic_inference(self):
        """Test evaluation mode determinism."""
        embedder = BandChannelEmbedding(d_model=self.d_model, dropout=0.1)
        embedder.eval()

        out1 = embedder(
            self.sample_tokens,
            num_bands=self.num_bands,
            num_channels=self.num_channels,
        )
        out2 = embedder(
            self.sample_tokens,
            num_bands=self.num_bands,
            num_channels=self.num_channels,
        )
        self.assertTrue(torch.equal(out1, out2))

    def test_serialization(self):
        """Test torch.save() & torch.load() parity."""
        embedder = BandChannelEmbedding(d_model=self.d_model)
        embedder.eval()

        out_orig = embedder(
            self.sample_tokens,
            num_bands=self.num_bands,
            num_channels=self.num_channels,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = os.path.join(tmp_dir, "embedder.pt")
            torch.save(embedder.state_dict(), ckpt_path)

            loaded = BandChannelEmbedding(d_model=self.d_model)
            loaded.load_state_dict(torch.load(ckpt_path))
            loaded.eval()

            out_loaded = loaded(
                self.sample_tokens,
                num_bands=self.num_bands,
                num_channels=self.num_channels,
            )
            self.assertTrue(torch.equal(out_orig, out_loaded))

    def test_gradient_flow(self):
        """Test non-zero gradient flow."""
        embedder = BandChannelEmbedding(d_model=self.d_model)
        x = self.sample_tokens.clone().requires_grad_(True)

        out = embedder(
            x, num_bands=self.num_bands, num_channels=self.num_channels
        )
        loss = out.pow(2).sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())
        self.assertFalse(torch.all(x.grad == 0))

        for param in embedder.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                self.assertFalse(torch.isnan(param.grad).any())


if __name__ == "__main__":
    unittest.main()
