"""
Temporal Projection Module for EEG Token Feature Embedding.

Projects temporal sample dimensions S to d_model embedding dimensions.
Decoupled into common modules to allow future replacement with Temporal CNN,
Wavelets, or MLP projection layers.
"""

import torch
import torch.nn as nn


class TemporalProjection(nn.Module):
    """
    Projects token temporal sample vectors (S) to d_model embedding space.

    Input shape:  (..., S)
    Output shape: (..., d_model)
    """

    def __init__(self, in_features: int, d_model: int = 128):
        super().__init__()
        self.in_features = in_features
        self.d_model = d_model
        self.projection = nn.Linear(in_features, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor ending in sample dimension S, e.g. (B, N, S)

        Returns:
            Projected tensor of shape (B, N, d_model)
        """
        if x.size(-1) != self.in_features:
            # Dynamically adapt linear layer if sample length changes
            self.in_features = x.size(-1)
            self.projection = nn.Linear(self.in_features, self.d_model).to(
                device=x.device, dtype=x.dtype
            )

        return self.projection(x)
