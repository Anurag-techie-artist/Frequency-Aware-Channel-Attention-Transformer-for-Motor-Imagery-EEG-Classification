"""
Band & Channel Positional Embedding Module for Frequency-Aware Transformer Encoder.

Projects token sample vectors S -> d_model and adds learnable Band and Channel
hierarchical positional embeddings.

Formulation:
    E_k = Proj(S_k) + BandEmbedding[f] + ChannelEmbedding[c]
    where k = f * num_channels + c
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from models.common.temporal_projection import TemporalProjection


@dataclass
class EmbeddingConfig:
    """Configuration options for BandChannelEmbedding."""

    d_model: int = 128
    dropout: float = 0.1
    max_bands: int = 32
    max_channels: int = 512


class BandChannelEmbedding(nn.Module):
    """
    Band x Channel Hierarchical Positional Embedding.

    Applies temporal sample projection and adds learnable frequency band and
    channel positional embeddings.
    """

    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        d_model: Optional[int] = None,
        dropout: Optional[float] = None,
        num_bands: Optional[int] = None,
        num_channels: Optional[int] = None,
    ):
        super().__init__()
        self.config = config or EmbeddingConfig()

        if d_model is not None:
            self.config.d_model = d_model
        if dropout is not None:
            self.config.dropout = dropout

        self.d_model = self.config.d_model
        self.num_bands = num_bands or 4
        self.num_channels = num_channels or 133

        # Temporal sample projection module S -> d_model
        self.temporal_projection = TemporalProjection(
            in_features=250, d_model=self.d_model
        )

        # Learnable Band & Channel Positional Embeddings
        self.max_bands = max(self.config.max_bands, self.num_bands)
        self.max_channels = max(self.config.max_channels, self.num_channels)

        self.band_embedding = nn.Embedding(self.max_bands, self.d_model)
        self.channel_embedding = nn.Embedding(self.max_channels, self.d_model)

        self.dropout = nn.Dropout(p=self.config.dropout)

    def forward(
        self,
        x_tokens: torch.Tensor,
        num_bands: int,
        num_channels: int,
    ) -> torch.Tensor:
        """
        Embed token sequence x_tokens.

        Args:
            x_tokens: Token sequence tensor of shape (B, N, S) or (N, S)
            num_bands: Number of frequency bands F
            num_channels: Number of EEG channels C

        Returns:
            Embedded token sequence of shape (B, N, d_model) or (N, d_model)
        """
        unbatched = False
        if x_tokens.dim() == 2:
            unbatched = True
            x_tokens = x_tokens.unsqueeze(0)
        elif x_tokens.dim() != 3:
            raise ValueError(
                f"BandChannelEmbedding expects 2D or 3D tensor, got shape {tuple(x_tokens.shape)}"
            )

        batch_size, num_tokens, num_samples = x_tokens.shape
        expected_tokens = num_bands * num_channels
        if num_tokens != expected_tokens:
            raise ValueError(
                f"Token count mismatch: expected {expected_tokens} (F={num_bands} * C={num_channels}), got {num_tokens}"
            )

        # Ensure embedding tables cover required band/channel counts
        if num_bands > self.band_embedding.num_embeddings:
            self.band_embedding = nn.Embedding(num_bands, self.d_model).to(
                device=x_tokens.device, dtype=x_tokens.dtype
            )
        if num_channels > self.channel_embedding.num_embeddings:
            self.channel_embedding = nn.Embedding(num_channels, self.d_model).to(
                device=x_tokens.device, dtype=x_tokens.dtype
            )

        # 1. Project temporal samples S -> d_model: (B, N, S) -> (B, N, d_model)
        projected = self.temporal_projection(x_tokens)

        # 2. Compute token band & channel positional indices for k in 0..N-1
        # Token ordering k = f * num_channels + c
        token_indices = torch.arange(num_tokens, device=x_tokens.device)
        band_indices = torch.div(token_indices, num_channels, rounding_mode="floor")
        channel_indices = token_indices % num_channels

        # 3. Retrieve positional embeddings
        b_embed = self.band_embedding(band_indices)  # (N, d_model)
        c_embed = self.channel_embedding(channel_indices)  # (N, d_model)

        # 4. Sum projected features + Band Positional + Channel Positional
        pos_embed = (b_embed + c_embed).unsqueeze(0)  # (1, N, d_model)
        embeddings = projected + pos_embed

        out = self.dropout(embeddings)
        return out.squeeze(0) if unbatched else out
