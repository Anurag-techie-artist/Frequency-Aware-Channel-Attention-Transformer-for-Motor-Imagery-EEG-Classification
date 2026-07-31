"""
Classification Head for Motor Imagery EEG Classification.

Converts global CLS embedding (B, d_model) into raw class logits (B, num_classes).

Architecture:
    CLS (B, d_model)
        │
        ▼
    LayerNorm(d_model)
        │
        ▼
    Linear(d_model -> hidden_dim)
        │
        ▼
    GELU Activation
        │
        ▼
    Dropout(p)
        │
        ▼
    Linear(hidden_dim -> num_classes)
        │
        ▼
    Logits (B, num_classes) [No Softmax]
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn


@dataclass
class ClassificationHeadConfig:
    """Configuration options for ClassificationHead."""

    d_model: int = 128
    num_classes: int = 4
    hidden_dim: int = 256
    dropout: float = 0.3
    activation: str = "gelu"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ClassificationHeadConfig":
        """Build config from dictionary."""
        return cls(
            d_model=int(d.get("d_model", 128)),
            num_classes=int(d.get("num_classes", 4)),
            hidden_dim=int(d.get("hidden_dim", 256)),
            dropout=float(d.get("dropout", 0.3)),
            activation=str(d.get("activation", "gelu")),
        )


class ClassificationHead(nn.Module):
    """
    MLP Classification Head module.

    Maps global CLS embedding representation to unnormalized class logits.
    """

    def __init__(
        self,
        config: Optional[ClassificationHeadConfig] = None,
        d_model: Optional[int] = None,
        num_classes: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        dropout: Optional[float] = None,
        activation: Optional[str] = None,
    ):
        super().__init__()
        base_cfg = config or ClassificationHeadConfig()

        self.config = ClassificationHeadConfig(
            d_model=d_model if d_model is not None else base_cfg.d_model,
            num_classes=num_classes if num_classes is not None else base_cfg.num_classes,
            hidden_dim=hidden_dim if hidden_dim is not None else base_cfg.hidden_dim,
            dropout=dropout if dropout is not None else base_cfg.dropout,
            activation=activation if activation is not None else base_cfg.activation,
        )

        act_str = self.config.activation.lower()
        if act_str == "gelu":
            act_layer = nn.GELU()
        elif act_str == "relu":
            act_layer = nn.ReLU()
        elif act_str == "silu":
            act_layer = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {self.config.activation}")

        self.mlp = nn.Sequential(
            nn.LayerNorm(self.config.d_model),
            nn.Linear(self.config.d_model, self.config.hidden_dim),
            act_layer,
            nn.Dropout(p=self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass converting CLS embedding to raw logits.

        Args:
            x: CLS embedding tensor of shape (B, d_model) or (d_model,)

        Returns:
            Logits tensor of shape (B, num_classes) or (num_classes,)
        """
        unbatched = False
        if x.dim() == 1:
            unbatched = True
            x = x.unsqueeze(0)
        elif x.dim() != 2:
            raise ValueError(
                f"ClassificationHead expects 1D or 2D tensor, got shape {tuple(x.shape)}"
            )

        logits = self.mlp(x)
        return logits.squeeze(0) if unbatched else logits
