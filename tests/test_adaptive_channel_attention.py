"""
Unit Tests for Adaptive Channel Attention (ACA) Module (Phase 4).

Tests:
1. test_shape_preservation: Verify output tensor shapes match input for 3D and 4D tensors.
2. test_exact_identity_disabled: Verify enabled=False returns exact torch.equal identity.
3. test_residual_connection: Verify residual vs non-residual scaling behavior.
4. test_attention_weight_bounds: Verify attention weights strictly in range [0.0, 1.0].
5. test_gradient_flow: Verify loss backward pass computes non-zero, non-NaN gradients.
6. test_eval_mode_deterministic: Verify evaluation mode produces identical outputs.
7. test_batch_size_invariance: Verify processing in batch vs individual samples matches.
8. test_variable_channels_and_bands: Test dynamic adaptability for various (F, C) shapes.
9. test_return_attention_api: Test return_attention=True and AttentionOutput dataclass.
10. test_serialization: Verify torch.save() and torch.load() produce identical results.
11. test_numerical_stability: Verify no NaN/Inf outputs under edge-case inputs.
12. test_device_compatibility: Test CPU (and GPU if available).
13. test_aca_alias: Verify QoL alias ACA.
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

from models.attention import (
    AdaptiveChannelAttention,
    ACA,
    AdaptiveChannelAttentionConfig,
    AttentionOutput,
    AttentionMetadata,
)


class TestAdaptiveChannelAttention(unittest.TestCase):
    """Test suite for AdaptiveChannelAttention (ACA) PyTorch module."""

    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 4
        self.num_bands = 4
        self.num_channels = 133
        self.num_samples = 250

        # Sample 4D EEG frequency tensor (B, Bands, Channels, Samples)
        self.sample_batch = torch.randn(
            self.batch_size, self.num_bands, self.num_channels, self.num_samples
        )

        # Sample 3D unbatched tensor (Bands, Channels, Samples)
        self.sample_single = torch.randn(
            self.num_bands, self.num_channels, self.num_samples
        )

    def test_aca_alias(self):
        """Test that QoL alias ACA points to AdaptiveChannelAttention."""
        self.assertIs(ACA, AdaptiveChannelAttention)

    def test_shape_preservation(self):
        """Test that ACA preserves input shapes exactly for 3D and 4D tensors."""
        aca = AdaptiveChannelAttention(
            num_channels=self.num_channels, num_bands=self.num_bands
        )

        out_4d = aca(self.sample_batch)
        self.assertEqual(out_4d.shape, self.sample_batch.shape)

        out_3d = aca(self.sample_single)
        self.assertEqual(out_3d.shape, self.sample_single.shape)

    def test_exact_identity_disabled(self):
        """Test that enabled=False returns exact torch.equal identity."""
        aca = AdaptiveChannelAttention(enabled=False)
        out = aca(self.sample_batch)
        self.assertTrue(torch.equal(self.sample_batch, out))

        out_ret, att_out = aca(self.sample_batch, return_attention=True)
        self.assertTrue(torch.equal(self.sample_batch, out_ret))
        self.assertIsInstance(att_out, AttentionOutput)

    def test_residual_connection(self):
        """Test residual scaling (Y = X * (1 + w)) vs non-residual (Y = X * w)."""
        aca_res = AdaptiveChannelAttention(residual=True, dropout=0.0)
        aca_res.eval()
        out_res, att_res = aca_res(self.sample_batch, return_attention=True)
        w_res = att_res.attention_weights.unsqueeze(-1)
        expected_res = self.sample_batch * (1.0 + w_res)
        self.assertTrue(torch.allclose(out_res, expected_res, atol=1e-5))

        aca_no_res = AdaptiveChannelAttention(residual=False, dropout=0.0)
        aca_no_res.eval()
        out_no_res, att_no_res = aca_no_res(self.sample_batch, return_attention=True)
        w_no_res = att_no_res.attention_weights.unsqueeze(-1)
        expected_no_res = self.sample_batch * w_no_res
        self.assertTrue(torch.allclose(out_no_res, expected_no_res, atol=1e-5))

    def test_attention_weight_bounds(self):
        """Test that attention weights strictly lie in range [0.0, 1.0]."""
        aca = AdaptiveChannelAttention(num_channels=self.num_channels)
        _, att_out = aca(self.sample_batch, return_attention=True)
        w = att_out.attention_weights
        self.assertTrue(torch.all(w >= 0.0))
        self.assertTrue(torch.all(w <= 1.0))

    def test_gradient_flow(self):
        """Test backward pass produces non-zero, non-NaN gradients."""
        aca = AdaptiveChannelAttention(num_channels=self.num_channels)
        x = self.sample_batch.clone().requires_grad_(True)
        out = aca(x)
        loss = out.pow(2).sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())
        self.assertFalse(torch.all(x.grad == 0))

        # Ensure module parameters have non-zero gradients
        for param in aca.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                self.assertFalse(torch.isnan(param.grad).any())

    def test_eval_mode_deterministic(self):
        """Test that model.eval() produces exact identical output tensors across calls."""
        aca = AdaptiveChannelAttention(dropout=0.5)
        aca.eval()
        out1 = aca(self.sample_batch)
        out2 = aca(self.sample_batch)
        self.assertTrue(torch.equal(out1, out2))

    def test_batch_size_invariance(self):
        """Test that processing samples individually vs in batch yields identical weights."""
        aca = AdaptiveChannelAttention(dropout=0.0)
        aca.eval()

        # Batch forward
        _, att_batch = aca(self.sample_batch, return_attention=True)
        w_batch = att_batch.attention_weights  # (B, F, C)

        # Individual forwards
        w_singles = []
        for i in range(self.batch_size):
            _, att_single = aca(self.sample_batch[i], return_attention=True)
            w_singles.append(att_single.attention_weights)
        w_singles = torch.stack(w_singles, dim=0)

        self.assertTrue(torch.allclose(w_batch, w_singles, atol=1e-5))

    def test_variable_channels_and_bands(self):
        """Test adaptability for various (F, C) shapes."""
        test_shapes = [
            (1, 16, 100),
            (2, 64, 200),
            (4, 133, 250),
            (8, 32, 128),
        ]
        for bands, chans, samps in test_shapes:
            with self.subTest(bands=bands, channels=chans, samples=samps):
                aca = AdaptiveChannelAttention()
                x = torch.randn(2, bands, chans, samps)
                out, att_out = aca(x, return_attention=True)
                self.assertEqual(out.shape, x.shape)
                self.assertEqual(att_out.attention_weights.shape, (2, bands, chans))
                self.assertEqual(att_out.metadata.num_bands, bands)
                self.assertEqual(att_out.metadata.num_channels, chans)

    def test_return_attention_api(self):
        """Test return_attention=True API and frozen AttentionOutput fields."""
        aca = AdaptiveChannelAttention()
        out, att_out = aca(self.sample_batch, return_attention=True)

        self.assertIsInstance(att_out, AttentionOutput)
        self.assertIsInstance(att_out.metadata, AttentionMetadata)
        self.assertEqual(att_out.features.shape, self.sample_batch.shape)
        self.assertEqual(
            att_out.attention_weights.shape,
            (self.batch_size, self.num_bands, self.num_channels),
        )

        # Immutability check
        with self.assertRaises(Exception):
            att_out.features = None  # Dataclass is frozen

    def test_serialization(self):
        """Test torch.save() and torch.load() output equivalence."""
        aca = AdaptiveChannelAttention(num_channels=self.num_channels)
        aca.eval()
        out_orig = aca(self.sample_batch)

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = os.path.join(tmp_dir, "aca.pt")
            torch.save(aca.state_dict(), ckpt_path)

            aca_loaded = AdaptiveChannelAttention(num_channels=self.num_channels)
            aca_loaded.load_state_dict(torch.load(ckpt_path))
            aca_loaded.eval()

            out_loaded = aca_loaded(self.sample_batch)
            self.assertTrue(torch.equal(out_orig, out_loaded))

    def test_numerical_stability(self):
        """Test zero NaN/Inf outputs under extreme float inputs."""
        aca = AdaptiveChannelAttention(num_channels=self.num_channels)

        zeros_input = torch.zeros_like(self.sample_batch)
        out_zeros = aca(zeros_input)
        self.assertFalse(torch.isnan(out_zeros).any())
        self.assertFalse(torch.isinf(out_zeros).any())

        large_input = torch.randn_like(self.sample_batch) * 1e5
        out_large = aca(large_input)
        self.assertFalse(torch.isnan(out_large).any())
        self.assertFalse(torch.isinf(out_large).any())

        tiny_input = torch.randn_like(self.sample_batch) * 1e-7
        out_tiny = aca(tiny_input)
        self.assertFalse(torch.isnan(out_tiny).any())
        self.assertFalse(torch.isinf(out_tiny).any())

    def test_device_compatibility(self):
        """Test CPU and CUDA (if available) device placement."""
        aca = AdaptiveChannelAttention(num_channels=self.num_channels)

        # CPU test
        out_cpu = aca(self.sample_batch)
        self.assertEqual(out_cpu.device.type, "cpu")

        # CUDA test if available
        if torch.cuda.is_available():
            aca_cuda = AdaptiveChannelAttention(num_channels=self.num_channels).cuda()
            batch_cuda = self.sample_batch.cuda()
            out_cuda = aca_cuda(batch_cuda)
            self.assertEqual(out_cuda.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
