"""
CLS Token Injection Module.

Maintains a learnable CLS token parameter of shape (1, 1, d_model) prepended to
token sequence embeddings (B, N, d_model) -> (B, N+1, d_model).
"""

from typing import Optional
import torch
import torch.nn as nn


class CLSToken(nn.Module):
    """
    Learnable CLS Token Layer.

    Prepends a learnable CLS token to input sequence embeddings.
    Input shape:  (B, N, d_model) or (N, d_model)
    Output shape: (B, N+1, d_model) or (N+1, d_model)
    """

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.d_model = d_model
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prepend CLS token to x.

        Args:
            x: Embedded sequence tensor of shape (B, N, d_model) or (N, d_model)

        Returns:
            Sequence tensor with CLS token prepended at index 0: (B, N+1, d_model)
        """
        unbatched = False
        if x.dim() == 2:
            unbatched = True
            x = x.unsqueeze(0)
        elif x.dim() != 3:
            raise ValueError(f"CLSToken expects 2D or 3D tensor, got shape {tuple(x.shape)}")

        batch_size = x.size(0)

        # Expand CLS token to match batch size: (1, 1, d_model) -> (B, 1, d_model)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # Prepend CLS token at index 0 along sequence dimension (dim 1)
        out = torch.cat([cls_tokens, x], dim=1)

        return out.squeeze(0) if unbatched else out
