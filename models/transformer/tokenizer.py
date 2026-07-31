"""
Band x Channel Tokenizer for Frequency-Aware Transformer Encoder (FATE).

Converts multi-band EEG tensors of shape (Batch, Bands, Channels, Samples) into
flat sequence token tensors of shape (Batch, Tokens, Samples) where
Tokens = Bands x Channels.

Token Index Ordering:
    Token Index k = f * Channels + c
    Corresponding to Frequency Band f and Channel c.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

torch_import_err = None
try:
    import torch
    import torch.nn as nn
except ImportError as e:
    torch_import_err = e


@dataclass(frozen=True)
class TokenMapping:
    """Immutable mapping from token index to (band, channel) identifiers."""

    token_index: int
    band_index: int
    channel_index: int
    band_name: str
    channel_name: str


@dataclass(frozen=True)
class TokenizationMetadata:
    """Immutable container holding token mapping registry and dimension metadata."""

    mappings: Tuple[TokenMapping, ...]
    num_tokens: int
    num_bands: int
    num_channels: int
    num_samples: int
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]


@dataclass
class TokenizerConfig:
    """Configuration options for BandChannelTokenizer."""

    order: str = "band_major"  # band_major: k = f * C + c


class BandChannelTokenizer(nn.Module):
    """
    Band x Channel Tokenizer.

    Transforms 4D tensors (Batch, Bands, Channels, Samples) to 3D token sequences
    (Batch, Bands * Channels, Samples) with deterministic token indexing.
    """

    def __init__(self, config: Optional[TokenizerConfig] = None):
        super().__init__()
        self.config = config or TokenizerConfig()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape input multi-band tensor into sequence of Band-Channel tokens.

        Args:
            x: Input tensor of shape (B, F, C, S) or (F, C, S)

        Returns:
            Token sequence tensor of shape (B, F*C, S) or (F*C, S)
        """
        unbatched = False
        if x.dim() == 3:
            unbatched = True
            x = x.unsqueeze(0)
        elif x.dim() != 4:
            raise ValueError(
                f"BandChannelTokenizer expects 3D or 4D tensor, got shape {tuple(x.shape)}"
            )

        batch_size, num_bands, num_channels, num_samples = x.shape
        num_tokens = num_bands * num_channels

        # Reshape to (B, F*C, S) preserving token order k = f * C + c
        tokens = x.reshape(batch_size, num_tokens, num_samples)

        return tokens.squeeze(0) if unbatched else tokens

    @staticmethod
    def get_metadata(
        num_bands: int,
        num_channels: int,
        num_samples: int,
        band_names: Optional[List[str]] = None,
        channel_names: Optional[List[str]] = None,
        batch_size: int = 1,
    ) -> TokenizationMetadata:
        """
        Generate deterministic TokenizationMetadata without coupling tensor execution.

        Token index ordering: k = f * num_channels + c
        """
        if band_names is None:
            band_names = [f"Band_{f}" for f in range(num_bands)]
        if channel_names is None:
            channel_names = [f"Ch_{c:03d}" for c in range(num_channels)]

        mappings: List[TokenMapping] = []
        token_k = 0
        for f in range(num_bands):
            b_name = band_names[f] if f < len(band_names) else f"Band_{f}"
            for c in range(num_channels):
                c_name = channel_names[c] if c < len(channel_names) else f"Ch_{c:03d}"
                mapping = TokenMapping(
                    token_index=token_k,
                    band_index=f,
                    channel_index=c,
                    band_name=b_name,
                    channel_name=c_name,
                )
                mappings.append(mapping)
                token_k += 1

        num_tokens = num_bands * num_channels
        return TokenizationMetadata(
            mappings=tuple(mappings),
            num_tokens=num_tokens,
            num_bands=num_bands,
            num_channels=num_channels,
            num_samples=num_samples,
            input_shape=(batch_size, num_bands, num_channels, num_samples),
            output_shape=(batch_size, num_tokens, num_samples),
        )
