"""
Loss Function Factory for Training Framework.
"""

from typing import Dict, Any
import torch.nn as nn


def build_loss(config: Dict[str, Any]) -> nn.Module:
    """
    Build loss criterion module from configuration dictionary.

    Args:
        config: Master or loss configuration dictionary

    Returns:
        PyTorch nn.Module loss criterion
    """
    loss_cfg = config.get("loss", {})
    loss_type = loss_cfg.get("type", "cross_entropy").lower()

    if loss_type in ["cross_entropy", "ce"]:
        label_smoothing = float(loss_cfg.get("label_smoothing", 0.0))
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    return nn.CrossEntropyLoss()
