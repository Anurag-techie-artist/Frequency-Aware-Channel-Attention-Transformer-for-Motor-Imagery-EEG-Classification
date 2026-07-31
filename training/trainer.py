"""
Trainer Module for EEGMotorImageryModel.

Orchestrates training epochs, validation epochs, AMP mixed precision, gradient clipping,
metric calculation, checkpoint saving/resuming, and experiment logging.
"""

import os
import logging
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.state import TrainState
from training.device import get_device
from training.checkpoint import CheckpointManager
from metrics import compute_classification_metrics
from loggers.experiment_logger import ExperimentLogger

logger = logging.getLogger(__name__)


class Trainer:
    """Model-agnostic Trainer for executing model training, validation, and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        exp_logger: Optional[ExperimentLogger] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config or {}
        train_cfg = self.config.get("training", {})

        self.device = device or get_device(train_cfg.get("device", "auto"))
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.epochs = int(train_cfg.get("epochs", 100))
        self.gradient_clip = float(train_cfg.get("gradient_clip", 1.0))
        self.use_amp = bool(train_cfg.get("mixed_precision", True)) and (self.device.type == "cuda")

        # AMP GradScaler if CUDA available
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        # Checkpoint manager
        ckpt_cfg = self.config.get("checkpoint", {})
        save_dir = ckpt_cfg.get("save_dir", "outputs/checkpoints")
        save_best = bool(ckpt_cfg.get("save_best", True))
        monitor = str(ckpt_cfg.get("monitor", "val_accuracy"))
        mode = str(ckpt_cfg.get("mode", "max"))

        self.checkpoint_manager = CheckpointManager(
            save_dir=save_dir, save_best=save_best, monitor=monitor, mode=mode
        )

        # Logger
        self.exp_logger = exp_logger
        self.state = TrainState(config=self.config)

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Execute a single training epoch across batches."""
        self.model.train()
        total_loss = 0.0
        all_logits = []
        all_targets = []

        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()

            if self.scaler:
                with torch.cuda.amp.autocast():
                    logits = self.model(x_batch)
                    loss = self.criterion(logits, y_batch)

                self.scaler.scale(loss).backward()
                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(x_batch)
                loss = self.criterion(logits, y_batch)
                loss.backward()
                if self.gradient_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()

            total_loss += loss.item() * x_batch.size(0)
            all_logits.append(logits.detach().cpu())
            all_targets.append(y_batch.detach().cpu())
            self.state.global_step += 1

        avg_loss = total_loss / len(dataloader.dataset)
        cat_logits = torch.cat(all_logits, dim=0)
        cat_targets = torch.cat(all_targets, dim=0)

        metrics = compute_classification_metrics(cat_logits, cat_targets)
        metrics["loss"] = avg_loss
        return metrics

    def validate_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Execute validation epoch across batches."""
        self.model.eval()
        total_loss = 0.0
        all_logits = []
        all_targets = []

        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits = self.model(x_batch)
                loss = self.criterion(logits, y_batch)

                total_loss += loss.item() * x_batch.size(0)
                all_logits.append(logits.cpu())
                all_targets.append(y_batch.cpu())

        avg_loss = total_loss / len(dataloader.dataset)
        cat_logits = torch.cat(all_logits, dim=0)
        cat_targets = torch.cat(all_targets, dim=0)

        metrics = compute_classification_metrics(cat_logits, cat_targets)
        metrics["loss"] = avg_loss
        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        resume_path: Optional[str] = None,
    ) -> TrainState:
        """
        Execute full training loop over epochs.

        Args:
            train_loader: DataLoader for training set
            val_loader: DataLoader for validation set
            resume_path: Optional checkpoint path to resume training from

        Returns:
            Updated TrainState object
        """
        if resume_path and os.path.exists(resume_path):
            self.state = self.checkpoint_manager.load_checkpoint(
                checkpoint_path=resume_path,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                device=self.device,
            )
            logger.info(f"Resumed training from epoch {self.state.epoch}")

        start_epoch = self.state.epoch + 1
        for epoch in range(start_epoch, self.epochs + 1):
            self.state.epoch = epoch

            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate_epoch(val_loader)

            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get("accuracy", 0.0))
                else:
                    self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            # Merge epoch metrics for logging
            epoch_log = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_f1": train_metrics["f1"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
                "learning_rate": current_lr,
            }

            if self.exp_logger:
                self.exp_logger.log_metrics(epoch_log, epoch=epoch)

            # Save checkpoints
            val_acc = val_metrics["accuracy"]
            self.checkpoint_manager.save_if_best(
                model=self.model,
                train_state=self.state,
                current_metric=val_acc,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
            )

            # Save latest checkpoint
            self.checkpoint_manager.save_checkpoint(
                model=self.model,
                train_state=self.state,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                filename="latest.pt",
            )

            logger.info(
                f"Epoch {epoch:03d}/{self.epochs:03d} | "
                f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} | "
                f"LR: {current_lr:.6f}"
            )

        return self.state
