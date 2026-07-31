"""
Adaptive Channel Attention (ACA) package.

Provides the AdaptiveChannelAttention module and associated dataclasses for
frequency-aware channel feature refinement.
"""

from models.attention.adaptive_channel_attention import (
    AdaptiveChannelAttention,
    ACA,
    AdaptiveChannelAttentionConfig,
    AttentionOutput,
    AttentionMetadata,
)

__all__ = [
    "AdaptiveChannelAttention",
    "ACA",
    "AdaptiveChannelAttentionConfig",
    "AttentionOutput",
    "AttentionMetadata",
]
