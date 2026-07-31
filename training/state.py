"""
TrainState Dataclass.

Encapsulates complete mutable training execution state (epoch, step, model weights,
optimizer states, scheduler states, scaler states, best metric, config).
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer


@dataclass
class TrainState:
    """Encapsulates training state parameters."""

    epoch: int = 0
    global_step: int = 0
    best_metric: float = 0.0

    model_state: Optional[Dict[str, Any]] = None
    optimizer_state: Optional[Dict[str, Any]] = None
    scheduler_state: Optional[Dict[str, Any]] = None
    scaler_state: Optional[Dict[str, Any]] = None

    config: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to serializable checkpoint dictionary."""
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "model_state": self.model_state,
            "optimizer_state": self.optimizer_state,
            "scheduler_state": self.scheduler_state,
            "scaler_state": self.scaler_state,
            "config": self.config,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainState":
        """Reconstruct TrainState from checkpoint dictionary."""
        return cls(
            epoch=int(d.get("epoch", 0)),
            global_step=int(d.get("global_step", 0)),
            best_metric=float(d.get("best_metric", 0.0)),
            model_state=d.get("model_state"),
            optimizer_state=d.get("optimizer_state"),
            scheduler_state=d.get("scheduler_state"),
            scaler_state=d.get("scaler_state"),
            config=d.get("config"),
        )
