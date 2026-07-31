"""
ExperimentRunner Orchestrator Module.

Loads configurations, sets reproducibility seed, resolves device, builds dataset DataLoaders,
instantiates EEGMotorImageryModel, configures optimizer/loss/scheduler, initializes experiment logger,
and executes Trainer.
"""

import os
import logging
from typing import Dict, Any, Optional

from configs.config_loader import load_master_config
from training.seed import set_seed
from training.device import get_device
from training.losses import build_loss
from training.optimizers import build_optimizer
from training.schedulers import build_scheduler
from training.trainer import Trainer
from models.eeg_motor_imagery_model import EEGMotorImageryModel
from datasets.builder import build_dataloaders
from loggers.experiment_logger import ExperimentLogger

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Orchestrates end-to-end experiment setup and training run execution."""

    def __init__(self, train_config_path: Optional[str] = None):
        self.config = load_master_config(train_cfg_path=train_config_path)

    def run(self, resume_path: Optional[str] = None):
        """Execute experiment run."""
        train_cfg = self.config.get("training", {})
        seed = int(train_cfg.get("seed", 42))
        set_seed(seed)

        device = get_device(train_cfg.get("device", "auto"))
        logger.info(f"Resolved execution device: {device}")

        # Assemble Model from Config
        model = EEGMotorImageryModel.from_config(self.config)

        # Build Loss, Optimizer, Scheduler
        criterion = build_loss(self.config)
        optimizer = build_optimizer(model, self.config)
        scheduler = build_scheduler(optimizer, self.config)

        # Logger
        log_cfg = self.config.get("logging", {})
        log_dir = log_cfg.get("log_dir", "outputs/logs")
        exp_logger = ExperimentLogger(
            log_dir=log_dir,
            use_csv=bool(log_cfg.get("csv", True)),
            use_tensorboard=bool(log_cfg.get("tensorboard", True)),
        )

        # Datasets & DataLoaders
        train_loader, val_loader, _ = build_dataloaders(self.config)

        # Trainer
        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            config=self.config,
            exp_logger=exp_logger,
            device=device,
        )

        logger.info("Starting experiment training run...")
        train_state = trainer.fit(train_loader, val_loader, resume_path=resume_path)
        logger.info(f"Experiment finished successfully at epoch {train_state.epoch}.")
        exp_logger.close()
        return train_state
