"""
CheckpointManager Module for State Saving, Loading, and Resuming Training.

Saves full training state dictionaries including model weights, optimizer states,
scheduler states, AMP scaler states, step metrics, and configuration settings.
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
from training.state import TrainState

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages model checkpointing, best metric tracking, and state resumption."""

    def __init__(
        self,
        save_dir: str = "outputs/checkpoints",
        save_best: bool = True,
        monitor: str = "val_accuracy",
        mode: str = "max",
    ):
        self.save_dir = save_dir
        self.save_best = save_best
        self.monitor = monitor
        self.mode = mode.lower()
        self.best_metric = -float("inf") if self.mode == "max" else float("inf")

        os.makedirs(save_dir, exist_ok=True)

    def is_better(self, current: float, best: float) -> bool:
        """Check if current metric improves upon best metric."""
        if self.mode == "max":
            return current > best
        return current < best

    def save_checkpoint(
        self,
        model: nn.Module,
        train_state: TrainState,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        filename: str = "latest.pt",
    ) -> str:
        """Save full training state checkpoint."""
        filepath = os.path.join(self.save_dir, filename)

        # Update train_state dictionary representations
        train_state.model_state = model.state_dict()
        if optimizer:
            train_state.optimizer_state = optimizer.state_dict()
        if scheduler and hasattr(scheduler, "state_dict"):
            train_state.scheduler_state = scheduler.state_dict()
        if scaler and hasattr(scaler, "state_dict"):
            train_state.scaler_state = scaler.state_dict()

        ckpt_dict = train_state.to_dict()
        torch.save(ckpt_dict, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
        return filepath

    def save_if_best(
        self,
        model: nn.Module,
        train_state: TrainState,
        current_metric: float,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
    ) -> Tuple[bool, str]:
        """Save checkpoint if current metric improves best_metric."""
        improved = False
        best_filepath = ""

        if self.is_better(current_metric, self.best_metric):
            self.best_metric = current_metric
            train_state.best_metric = current_metric
            improved = True
            best_filepath = self.save_checkpoint(
                model=model,
                train_state=train_state,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                filename="best.pt",
            )

        return improved, best_filepath

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ) -> TrainState:
        """Load state dictionary from checkpoint into model and training objects."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found.")

        map_location = device if device else "cpu"
        ckpt_dict = torch.load(checkpoint_path, map_location=map_location)
        train_state = TrainState.from_dict(ckpt_dict)

        if train_state.model_state:
            model.load_state_dict(train_state.model_state)
        if optimizer and train_state.optimizer_state:
            optimizer.load_state_dict(train_state.optimizer_state)
        if scheduler and train_state.scheduler_state and hasattr(scheduler, "load_state_dict"):
            scheduler.load_state_dict(train_state.scheduler_state)
        if scaler and train_state.scaler_state and hasattr(scaler, "load_state_dict"):
            scaler.load_state_dict(train_state.scaler_state)

        if self.mode == "max":
            self.best_metric = max(self.best_metric, train_state.best_metric)
        else:
            self.best_metric = min(self.best_metric, train_state.best_metric)

        logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {train_state.epoch})")
        return train_state
