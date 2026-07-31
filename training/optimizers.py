"""
Optimizer Factory for Training Framework.
"""

from typing import Dict, Any
import torch
import torch.nn as nn
from torch.optim import Optimizer, AdamW, SGD, Adam


def build_optimizer(model: nn.Module, config: Dict[str, Any]) -> Optimizer:
    """
    Build optimizer from model parameters and configuration dictionary.

    Args:
        model: PyTorch nn.Module containing model parameters
        config: Master or optimizer configuration dictionary

    Returns:
        PyTorch Optimizer instance
    """
    train_cfg = config.get("training", {})
    opt_cfg = config.get("optimizer", {})

    lr = float(opt_cfg.get("learning_rate", train_cfg.get("learning_rate", 3e-4)))
    weight_decay = float(
        opt_cfg.get("weight_decay", train_cfg.get("weight_decay", 0.01))
    )
    opt_type = opt_cfg.get("type", "adamw").lower()

    if opt_type == "adamw":
        betas = opt_cfg.get("betas", (0.9, 0.999))
        return AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay, betas=tuple(betas)
        )
    elif opt_type == "adam":
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_type == "sgd":
        momentum = float(opt_cfg.get("momentum", 0.9))
        return SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {opt_type}")
