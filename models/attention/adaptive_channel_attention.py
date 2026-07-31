"""
Adaptive Channel Attention (ACA) Module for Motor Imagery EEG Classification.

Learns the relative importance of EEG channels dynamically for each frequency
band prior to the Transformer encoder.

Mathematical Formulation:
    1. Dual Temporal Aggregation (GAP & GMP):
        z_avg(b, f, c) = mean_s(X(b, f, c, s))
        z_max(b, f, c) = max_s(X(b, f, c, s))
    2. Frequency-Aware Bottleneck Excitation:
        a_avg = W_2 * Dropout(Activation(W_1 * z_avg))
        a_max = W_2 * Dropout(Activation(W_1 * z_max))
        a = a_avg + a_max
    3. Channel Weights & Gated Amplification:
        w = Sigmoid(a) in [0.0, 1.0]
        Y = X * (1 + w)  [Residual amplification scale range: [1.0, 2.0]]
"""

import os
import time
import yaml
import logging
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AttentionMetadata:
    """Immutable metadata container for attention module execution."""

    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    num_bands: int
    num_channels: int
    residual_enabled: bool
    execution_time_ms: float


@dataclass(frozen=True)
class AttentionOutput:
    """Immutable container holding refined features, attention weights, and execution metadata."""

    features: torch.Tensor
    attention_weights: torch.Tensor
    metadata: AttentionMetadata


@dataclass
class AdaptiveChannelAttentionConfig:
    """Configuration options for Adaptive Channel Attention (ACA)."""

    enabled: bool = True
    hidden_ratio: int = 4
    dropout: float = 0.1
    activation: str = "gelu"
    residual: bool = True

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AdaptiveChannelAttentionConfig":
        """Build config instance from dictionary."""
        return cls(
            enabled=bool(d.get("enabled", True)),
            hidden_ratio=int(d.get("hidden_ratio", 4)),
            dropout=float(d.get("dropout", 0.1)),
            activation=str(d.get("activation", "gelu")),
            residual=bool(d.get("residual", True)),
        )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "AdaptiveChannelAttentionConfig":
        """Load configuration from YAML file."""
        if not os.path.exists(yaml_path):
            logger.warning(f"Config file {yaml_path} not found. Using default ACA configuration.")
            return cls()

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # Support root 'model.attention' or direct 'attention'
        att_dict = data.get("model", {}).get("attention", {}) if "model" in data else data.get("attention", {})
        if not att_dict and "attention" not in data and "model" not in data:
            att_dict = data

        return cls.from_dict(att_dict)


class AdaptiveChannelAttention(nn.Module):
    """
    Adaptive Channel Attention (ACA) Module.

    Learns channel importance independently per frequency band without collapsing
    the frequency or temporal dimensions.

    Input shape:  (Batch, Bands, Channels, Samples)  or  (Bands, Channels, Samples)
    Output shape: Exact same dimension as input.
    """

    def __init__(
        self,
        config: Optional[AdaptiveChannelAttentionConfig] = None,
        num_channels: Optional[int] = None,
        num_bands: Optional[int] = None,
        hidden_ratio: Optional[int] = None,
        dropout: Optional[float] = None,
        activation: Optional[str] = None,
        residual: Optional[bool] = None,
        enabled: Optional[bool] = None,
    ):
        super().__init__()

        # Resolve configuration hierarchy (explicit kwargs override config object)
        if config is None:
            config = AdaptiveChannelAttentionConfig()

        self.config = AdaptiveChannelAttentionConfig(
            enabled=enabled if enabled is not None else config.enabled,
            hidden_ratio=hidden_ratio if hidden_ratio is not None else config.hidden_ratio,
            dropout=dropout if dropout is not None else config.dropout,
            activation=activation if activation is not None else config.activation,
            residual=residual if residual is not None else config.residual,
        )

        self.num_channels = num_channels
        self.num_bands = num_bands

        # Activation resolver
        act_str = self.config.activation.lower()
        if act_str == "gelu":
            self.activation = nn.GELU()
        elif act_str == "relu":
            self.activation = nn.ReLU()
        elif act_str == "silu" or act_str == "swish":
            self.activation = nn.SiLU()
        elif act_str == "leaky_relu":
            self.activation = nn.LeakyReLU(0.1)
        else:
            raise ValueError(f"Unsupported activation function: {self.config.activation}")

        self.dropout = nn.Dropout(p=self.config.dropout)

        # Bottleneck projection layers (initialized lazily if num_channels is None)
        if self.num_channels is not None:
            self._build_layers(self.num_channels)
        else:
            self.fc1 = None
            self.fc2 = None

    def _build_layers(self, num_channels: int):
        """Construct linear bottleneck projection layers."""
        self.num_channels = num_channels
        hidden_dim = max(1, num_channels // self.config.hidden_ratio)
        self.fc1 = nn.Linear(num_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_channels)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, AttentionOutput]]:
        """
        Forward pass for Adaptive Channel Attention module.

        Args:
            x: Input tensor of shape (B, Bands, Channels, Samples) or (Bands, Channels, Samples)
            return_attention: If True, returns tuple (features, AttentionOutput)

        Returns:
            Refined features tensor (same shape as input) if return_attention is False,
            otherwise tuple (refined_features, AttentionOutput).
        """
        start_time = time.perf_counter()

        # Handle unbatched input (F, C, S) -> (1, F, C, S)
        unbatched = False
        if x.dim() == 3:
            unbatched = True
            x = x.unsqueeze(0)
        elif x.dim() != 4:
            raise ValueError(
                f"AdaptiveChannelAttention expects 3D or 4D tensor, got shape {tuple(x.shape)}"
            )

        batch_size, num_bands, num_channels, num_samples = x.shape

        # If module disabled, return identity exactly
        if not self.config.enabled:
            output_tensor = x.squeeze(0) if unbatched else x
            if not return_attention:
                return output_tensor

            exec_time = (time.perf_counter() - start_time) * 1000.0
            dummy_weights = torch.ones(
                (batch_size, num_bands, num_channels), device=x.device, dtype=x.dtype
            )
            meta = AttentionMetadata(
                input_shape=tuple(x.shape),
                output_shape=tuple(x.shape),
                num_bands=num_bands,
                num_channels=num_channels,
                residual_enabled=self.config.residual,
                execution_time_ms=exec_time,
            )
            att_out = AttentionOutput(
                features=output_tensor,
                attention_weights=dummy_weights.squeeze(0) if unbatched else dummy_weights,
                metadata=meta,
            )
            return output_tensor, att_out

        # Dynamically build layers if input channels were not pre-specified
        if self.fc1 is None or self.num_channels != num_channels:
            # Move created layers to input tensor's device & dtype
            self._build_layers(num_channels)
            self.to(device=x.device, dtype=x.dtype)

        # 1. Dual Temporal Aggregation across sample dimension S (dim -1)
        z_avg = torch.mean(x, dim=-1)  # (B, F, C)
        z_max = torch.max(x, dim=-1).values  # (B, F, C)

        # 2. Bottleneck excitation (applied along channel dimension C per band F independently)
        # down-projection -> activation -> dropout -> up-projection
        a_avg = self.fc2(self.dropout(self.activation(self.fc1(z_avg))))  # (B, F, C)
        a_max = self.fc2(self.dropout(self.activation(self.fc1(z_max))))  # (B, F, C)

        # Fuse descriptors & apply Sigmoid activation
        a_fused = a_avg + a_max  # (B, F, C)
        attention_weights = torch.sigmoid(a_fused)  # (B, F, C) in range [0.0, 1.0]

        # 3. Apply attention scaling to input features
        w_expanded = attention_weights.unsqueeze(-1)  # (B, F, C, 1)

        if self.config.residual:
            # Residual amplification: Y = X + X * w = X * (1 + w) -> Range [1.0, 2.0]
            y = x * (1.0 + w_expanded)
        else:
            # Direct attention gating: Y = X * w -> Range [0.0, 1.0]
            y = x * w_expanded

        # Restore original tensor rank if input was 3D
        out_features = y.squeeze(0) if unbatched else y
        out_weights = attention_weights.squeeze(0) if unbatched else attention_weights

        if not return_attention:
            return out_features

        exec_time = (time.perf_counter() - start_time) * 1000.0
        meta = AttentionMetadata(
            input_shape=tuple(x.shape),
            output_shape=tuple(y.shape),
            num_bands=num_bands,
            num_channels=num_channels,
            residual_enabled=self.config.residual,
            execution_time_ms=exec_time,
        )
        attention_output = AttentionOutput(
            features=out_features,
            attention_weights=out_weights,
            metadata=meta,
        )
        return out_features, attention_output


# Quality-of-life API alias
ACA = AdaptiveChannelAttention
