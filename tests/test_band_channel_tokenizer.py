"""
Unit Tests for BandChannelTokenizer (Phase 5).

Tests:
1. test_reshape_correctness: (B, F, C, S) -> (B, F*C, S) for 3D and 4D tensors.
2. test_token_order_stability: Verify token 0 is strictly Band 0 / Ch 0 and ordering k = f*C + c is stable.
3. test_metadata_generation: Verify get_metadata produces correct mappings and metadata fields.
4. test_dynamic_shapes: Test arbitrary (F, C, S) tensor dimensions.
"""

import os
import sys
import unittest
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.transformer import (
    BandChannelTokenizer,
    TokenizerConfig,
    TokenizationMetadata,
    TokenMapping,
)


class TestBandChannelTokenizer(unittest.TestCase):
    """Test suite for BandChannelTokenizer module."""

    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 3
        self.num_bands = 4
        self.num_channels = 133
        self.num_samples = 250
        self.num_tokens = self.num_bands * self.num_channels

        self.sample_tensor = torch.randn(
            self.batch_size, self.num_bands, self.num_channels, self.num_samples
        )

    def test_reshape_correctness(self):
        """Test reshape transformation (B, F, C, S) -> (B, F*C, S)."""
        tokenizer = BandChannelTokenizer()

        tokens_4d = tokenizer(self.sample_tensor)
        self.assertEqual(
            tokens_4d.shape, (self.batch_size, self.num_tokens, self.num_samples)
        )

        sample_3d = self.sample_tensor[0]
        tokens_3d = tokenizer(sample_3d)
        self.assertEqual(tokens_3d.shape, (self.num_tokens, self.num_samples))

    def test_token_order_stability(self):
        """Test that token indexing k = f*C + c remains strictly deterministic across runs."""
        tokenizer = BandChannelTokenizer()

        # Run tokenization twice
        tokens1 = tokenizer(self.sample_tensor)
        tokens2 = tokenizer(self.sample_tensor)
        self.assertTrue(torch.equal(tokens1, tokens2))

        # Check indexing for sample values: band 0, channel 0 should equal token 0
        expected_token_0 = self.sample_tensor[:, 0, 0, :]  # (B, S)
        actual_token_0 = tokens1[:, 0, :]  # (B, S)
        self.assertTrue(torch.equal(expected_token_0, actual_token_0))

        # Check indexing for band 1, channel 5 -> token index 1 * 133 + 5 = 138
        k = 1 * self.num_channels + 5
        expected_token_k = self.sample_tensor[:, 1, 5, :]
        actual_token_k = tokens1[:, k, :]
        self.assertTrue(torch.equal(expected_token_k, actual_token_k))

    def test_metadata_generation(self):
        """Test get_metadata produces correct token mappings and immutable TokenizationMetadata."""
        meta = BandChannelTokenizer.get_metadata(
            num_bands=self.num_bands,
            num_channels=self.num_channels,
            num_samples=self.num_samples,
            band_names=["Theta", "Alpha", "Beta", "Gamma"],
            batch_size=self.batch_size,
        )

        self.assertIsInstance(meta, TokenizationMetadata)
        self.assertEqual(meta.num_tokens, 532)
        self.assertEqual(len(meta.mappings), 532)

        # Check first token
        first_map = meta.mappings[0]
        self.assertEqual(first_map.token_index, 0)
        self.assertEqual(first_map.band_index, 0)
        self.assertEqual(first_map.channel_index, 0)
        self.assertEqual(first_map.band_name, "Theta")

        # Check last token index 531: Band 3 (Gamma), Channel 132
        last_map = meta.mappings[531]
        self.assertEqual(last_map.token_index, 531)
        self.assertEqual(last_map.band_index, 3)
        self.assertEqual(last_map.channel_index, 132)
        self.assertEqual(last_map.band_name, "Gamma")

    def test_dynamic_shapes(self):
        """Test tokenizer with dynamic / non-standard tensor dimensions."""
        shapes = [
            (1, 16, 50),
            (2, 64, 100),
            (8, 32, 250),
        ]
        tokenizer = BandChannelTokenizer()
        for f, c, s in shapes:
            with self.subTest(bands=f, channels=c, samples=s):
                x = torch.randn(2, f, c, s)
                tokens = tokenizer(x)
                self.assertEqual(tokens.shape, (2, f * c, s))


if __name__ == "__main__":
    unittest.main()
