"""
Frequency-Aware Transformer Encoder (FATE) for Motor Imagery EEG Classification.

Combines BandChannelTokenizer, BandChannelEmbedding, and PyTorch TransformerEncoder
stack to process multi-band EEG token sequences.

Input Shape:  (Batch, Bands, Channels, Samples) -> (B, F, C, S)
Output Shape: Contextual Embeddings (Batch, Tokens, d_model) where Tokens = F * C.
"""

import os
import yaml
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional, Union, List

import torch
import torch.nn as nn

from models.transformer.tokenizer import (
    BandChannelTokenizer,
    TokenizerConfig,
    TokenizationMetadata,
)
from models.transformer.embeddings import BandChannelEmbedding, EmbeddingConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FATEOutput:
    """Immutable container holding contextual token embeddings and tokenization metadata."""

    contextual_embeddings: torch.Tensor
    token_metadata: TokenizationMetadata


@dataclass
class TransformerConfig:
    """Configuration options for TransformerEncoder stack."""

    d_model: int = 128
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    activation: str = "gelu"
    batch_first: bool = True


@dataclass
class FrequencyAwareTransformerConfig:
    """Configuration options for FrequencyAwareTransformer (FATE)."""

    enabled: bool = True
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FrequencyAwareTransformerConfig":
        """Load configuration from dictionary."""
        enabled = bool(d.get("enabled", True))
        t_dict = d.get("transformer", {})
        if not t_dict and "d_model" in d:
            t_dict = d

        t_config = TransformerConfig(
            d_model=int(t_dict.get("d_model", 128)),
            nhead=int(t_dict.get("nhead", 8)),
            num_layers=int(t_dict.get("num_layers", 4)),
            dim_feedforward=int(t_dict.get("dim_feedforward", 512)),
            dropout=float(t_dict.get("dropout", 0.1)),
            activation=str(t_dict.get("activation", "gelu")),
            batch_first=bool(t_dict.get("batch_first", True)),
        )
        e_config = EmbeddingConfig(
            d_model=t_config.d_model,
            dropout=t_config.dropout,
        )
        tok_config = TokenizerConfig()

        return cls(
            enabled=enabled,
            tokenizer=tok_config,
            embedding=e_config,
            transformer=t_config,
        )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "FrequencyAwareTransformerConfig":
        """Load configuration from YAML file."""
        if not os.path.exists(yaml_path):
            logger.warning(
                f"Config file {yaml_path} not found. Using default FATE configuration."
            )
            return cls()

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        model_dict = data.get("model", {}) if "model" in data else data
        trans_dict = model_dict.get("transformer", model_dict)
        return cls.from_dict(trans_dict)


class FrequencyAwareTransformer(nn.Module):
    """
    Frequency-Aware Transformer Encoder (FATE).

    Processes ACA output tensors (B, F, C, S) into contextualized token sequence
    embeddings (B, F*C, d_model) via multi-head self-attention.
    """

    def __init__(
        self,
        config: Optional[FrequencyAwareTransformerConfig] = None,
        d_model: Optional[int] = None,
        nhead: Optional[int] = None,
        num_layers: Optional[int] = None,
        dim_feedforward: Optional[int] = None,
        dropout: Optional[float] = None,
        activation: Optional[str] = None,
        enabled: Optional[bool] = None,
    ):
        super().__init__()
        base_cfg = config or FrequencyAwareTransformerConfig()

        # Allow explicit keyword argument overrides
        t_cfg = base_cfg.transformer
        d_model = d_model if d_model is not None else t_cfg.d_model
        nhead = nhead if nhead is not None else t_cfg.nhead
        num_layers = num_layers if num_layers is not None else t_cfg.num_layers
        dim_feedforward = (
            dim_feedforward if dim_feedforward is not None else t_cfg.dim_feedforward
        )
        dropout = dropout if dropout is not None else t_cfg.dropout
        activation = activation if activation is not None else t_cfg.activation
        enabled = enabled if enabled is not None else base_cfg.enabled

        self.config = FrequencyAwareTransformerConfig(
            enabled=enabled,
            tokenizer=base_cfg.tokenizer,
            embedding=EmbeddingConfig(d_model=d_model, dropout=dropout),
            transformer=TransformerConfig(
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                batch_first=True,
            ),
        )

        self.tokenizer = BandChannelTokenizer(config=self.config.tokenizer)
        self.embedding = BandChannelEmbedding(config=self.config.embedding)

        # PyTorch Standard TransformerEncoder Stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(
        self,
        x: torch.Tensor,
        return_metadata: bool = False,
        band_names: Optional[List[str]] = None,
        channel_names: Optional[List[str]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, FATEOutput]]:
        """
        Forward pass for FATE model.

        Args:
            x: Input tensor of shape (B, F, C, S) or (F, C, S)
            return_metadata: If True, returns tuple (contextual_embeddings, FATEOutput)
            band_names: Optional list of frequency band names
            channel_names: Optional list of channel names

        Returns:
            contextual_embeddings of shape (B, F*C, d_model) if return_metadata is False,
            otherwise tuple (contextual_embeddings, FATEOutput).
        """
        unbatched = False
        if x.dim() == 3:
            unbatched = True
            x = x.unsqueeze(0)
        elif x.dim() != 4:
            raise ValueError(
                f"FrequencyAwareTransformer expects 3D or 4D tensor, got shape {tuple(x.shape)}"
            )

        batch_size, num_bands, num_channels, num_samples = x.shape

        # 1. Tokenize (B, F, C, S) -> (B, F*C, S)
        tokens = self.tokenizer(x)

        # 2. Embed tokens with Linear Projection + 2D Positional Embeddings
        embedded = self.embedding(
            tokens, num_bands=num_bands, num_channels=num_channels
        )

        # 3. Process sequence tokens through TransformerEncoder
        contextual_embeddings = self.transformer_encoder(embedded)

        out_features = (
            contextual_embeddings.squeeze(0) if unbatched else contextual_embeddings
        )

        if not return_metadata:
            return out_features

        metadata = BandChannelTokenizer.get_metadata(
            num_bands=num_bands,
            num_channels=num_channels,
            num_samples=num_samples,
            band_names=band_names,
            channel_names=channel_names,
            batch_size=batch_size,
        )
        fate_output = FATEOutput(
            contextual_embeddings=out_features,
            token_metadata=metadata,
        )
        return out_features, fate_output


# Quality-of-life API alias
FATE = FrequencyAwareTransformer
