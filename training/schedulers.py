"""
Learning Rate Scheduler Factory for Training Framework.
"""

from typing import Dict, Any, Optional
import torch.optim as optim
from torch.optim.lr_scheduler import (
    _LRScheduler,
    CosineAnnealingLR,
    StepLR,
    ReduceLROnPlateau,
)


def build_scheduler(
    optimizer: optim.Optimizer, config: Dict[str, Any]
) -> Optional[Any]:
    """
    Build learning rate scheduler instance.

    Args:
        optimizer: Configured PyTorch Optimizer
        config: Master or scheduler configuration dictionary

    Returns:
        PyTorch _LRScheduler or ReduceLROnPlateau instance (or None if disabled)
    """
    sched_cfg = config.get("scheduler", {})
    train_cfg = config.get("training", {})

    sched_type = sched_cfg.get("type", "cosine").lower()
    epochs = int(train_cfg.get("epochs", 100))

    if sched_type == "cosine":
        min_lr = float(sched_cfg.get("min_lr", 1e-6))
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    elif sched_type == "step":
        step_size = int(sched_cfg.get("step_size", 30))
        gamma = float(sched_cfg.get("gamma", 0.1))
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif sched_type in ["plateau", "reduce_on_plateau"]:
        mode = sched_cfg.get("mode", "max")
        patience = int(sched_cfg.get("patience", 10))
        factor = float(sched_cfg.get("factor", 0.5))
        return ReduceLROnPlateau(
            optimizer, mode=mode, patience=patience, factor=factor
        )

    return None
