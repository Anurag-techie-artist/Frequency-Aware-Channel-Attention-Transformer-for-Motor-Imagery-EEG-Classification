"""
Unit Tests for FrequencyAwareTransformer / FATE (Phase 5).

Tests:
1. test_fate_alias: QoL alias FATE points to FrequencyAwareTransformer.
2. test_shape_preservation: Output shape (B, F*C, d_model) for 3D and 4D tensors.
3. test_variable_batch_sizes: B = 1, 4, 8.
4. test_variable_channels_and_bands: Arbitrary (F, C, S) shapes.
5. test_numerical_stability: Assert zero NaN or Inf values under extreme inputs.
6. test_gradient_flow: Backward pass updates weights cleanly.
7. test_serialization: Save and load parity via state_dict.
8. test_device_compatibility: CPU and CUDA placement.
9. test_deterministic_eval_mode: Model eval mode determinism.
10. test_return_metadata_api: Test return_metadata=True and FATEOutput dataclass.
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

from models.transformer import (
    FrequencyAwareTransformer,
    FATE,
    FATEOutput,
    FrequencyAwareTransformerConfig,
    TokenizationMetadata,
)


class TestFrequencyAwareTransformer(unittest.TestCase):
    """Test suite for FrequencyAwareTransformer (FATE) module."""

    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 2
        self.num_bands = 4
        self.num_channels = 133
        self.num_samples = 250
        self.d_model = 128
        self.num_tokens = self.num_bands * self.num_channels

        self.sample_tensor = torch.randn(
            self.batch_size, self.num_bands, self.num_channels, self.num_samples
        )

    def test_fate_alias(self):
        """Test QoL alias FATE is FrequencyAwareTransformer."""
        self.assertIs(FATE, FrequencyAwareTransformer)

    def test_shape_preservation(self):
        """Test output shape (B, N, d_model) where N = F * C."""
        fate = FATE(d_model=self.d_model)

        out_4d = fate(self.sample_tensor)
        self.assertEqual(
            out_4d.shape, (self.batch_size, self.num_tokens, self.d_model)
        )

        sample_3d = self.sample_tensor[0]
        out_3d = fate(sample_3d)
        self.assertEqual(out_3d.shape, (self.num_tokens, self.d_model))

    def test_variable_batch_sizes(self):
        """Test FATE with different batch sizes B=1, 4, 8."""
        fate = FATE(d_model=self.d_model)
        for b in [1, 4, 8]:
            with self.subTest(batch_size=b):
                x = torch.randn(b, self.num_bands, self.num_channels, self.num_samples)
                out = fate(x)
                self.assertEqual(out.shape, (b, self.num_tokens, self.d_model))

    def test_variable_channels_and_bands(self):
        """Test FATE with arbitrary (F, C, S) shapes."""
        test_shapes = [
            (1, 16, 100),
            (2, 64, 200),
            (4, 133, 250),
            (8, 32, 128),
        ]
        for bands, chans, samps in test_shapes:
            with self.subTest(bands=bands, channels=chans, samples=samps):
                fate = FATE(d_model=64, nhead=4, num_layers=2)
                x = torch.randn(2, bands, chans, samps)
                out = fate(x)
                self.assertEqual(out.shape, (2, bands * chans, 64))

    def test_numerical_stability(self):
        """Test zero NaN or Inf outputs under extreme inputs."""
        fate = FATE(d_model=self.d_model)

        zeros_x = torch.zeros_like(self.sample_tensor)
        out_zeros = fate(zeros_x)
        self.assertFalse(torch.isnan(out_zeros).any())
        self.assertFalse(torch.isinf(out_zeros).any())

        large_x = torch.randn_like(self.sample_tensor) * 1e4
        out_large = fate(large_x)
        self.assertFalse(torch.isnan(out_large).any())
        self.assertFalse(torch.isinf(out_large).any())

    def test_gradient_flow(self):
        """Test backward pass produces non-zero, non-NaN gradients."""
        fate = FATE(d_model=self.d_model)
        x = self.sample_tensor.clone().requires_grad_(True)

        out = fate(x)
        loss = out.pow(2).sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())
        self.assertFalse(torch.all(x.grad == 0))

        for param in fate.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                self.assertFalse(torch.isnan(param.grad).any())

    def test_serialization(self):
        """Test torch.save() & torch.load() parity."""
        fate = FATE(d_model=self.d_model)
        fate.eval()
        out_orig = fate(self.sample_tensor)

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = os.path.join(tmp_dir, "fate.pt")
            torch.save(fate.state_dict(), ckpt_path)

            loaded_fate = FATE(d_model=self.d_model)
            loaded_fate.load_state_dict(torch.load(ckpt_path))
            loaded_fate.eval()

            out_loaded = loaded_fate(self.sample_tensor)
            self.assertTrue(torch.equal(out_orig, out_loaded))

    def test_device_compatibility(self):
        """Test CPU and CUDA placement."""
        fate = FATE(d_model=self.d_model)
        out_cpu = fate(self.sample_tensor)
        self.assertEqual(out_cpu.device.type, "cpu")

        if torch.cuda.is_available():
            fate_cuda = FATE(d_model=self.d_model).cuda()
            out_cuda = fate_cuda(self.sample_tensor.cuda())
            self.assertEqual(out_cuda.device.type, "cuda")

    def test_deterministic_eval_mode(self):
        """Test evaluation mode determinism."""
        fate = FATE(d_model=self.d_model, dropout=0.2)
        fate.eval()

        out1 = fate(self.sample_tensor)
        out2 = fate(self.sample_tensor)
        self.assertTrue(torch.equal(out1, out2))

    def test_return_metadata_api(self):
        """Test return_metadata=True signature and FATEOutput dataclass."""
        fate = FATE(d_model=self.d_model)
        out_feats, fate_out = fate(self.sample_tensor, return_metadata=True)

        self.assertIsInstance(fate_out, FATEOutput)
        self.assertIsInstance(fate_out.token_metadata, TokenizationMetadata)
        self.assertEqual(
            fate_out.contextual_embeddings.shape,
            (self.batch_size, self.num_tokens, self.d_model),
        )
        self.assertEqual(fate_out.token_metadata.num_tokens, 532)

        # Dataclass immutability check
        with self.assertRaises(Exception):
            fate_out.contextual_embeddings = None


if __name__ == "__main__":
    unittest.main()
