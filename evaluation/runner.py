"""
EvaluationRunner Orchestrator Module.

Loads master config, resolves execution device, loads trained checkpoint, builds DataLoaders,
and invokes Evaluator to run evaluation.
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple

import torch

from configs.config_loader import load_master_config
from models.eeg_motor_imagery_model import EEGMotorImageryModel
from datasets.builder import build_dataloaders
from training.device import get_device
from evaluation.evaluator import Evaluator
from evaluation.inference import InferenceResults

logger = logging.getLogger(__name__)


class EvaluationRunner:
    """Orchestrates checkpoint loading and evaluation execution."""

    def __init__(
        self,
        checkpoint_path: str,
        train_config_path: Optional[str] = None,
        output_dir: str = "outputs/evaluation",
    ):
        self.checkpoint_path = checkpoint_path
        self.output_dir = output_dir
        self.config = load_master_config(train_cfg_path=train_config_path)

    def run(self, generate_plots: bool = True) -> Tuple[Dict[str, Any], InferenceResults]:
        """Execute evaluation run from checkpoint."""
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint_path}")

        train_cfg = self.config.get("training", {})
        device = get_device(train_cfg.get("device", "auto"))
        logger.info(f"Resolved execution device: {device}")

        # Assemble Model and load weights
        model = EEGMotorImageryModel.from_config(self.config)
        ckpt_dict = torch.load(self.checkpoint_path, map_location=device)

        model_state = (
            ckpt_dict.get("model_state", ckpt_dict)
            if isinstance(ckpt_dict, dict)
            else ckpt_dict
        )
        model.load_state_dict(model_state)
        model.eval()

        # Build DataLoader
        _, val_loader, test_loader = build_dataloaders(self.config)
        eval_loader = test_loader or val_loader

        # Evaluator
        evaluator = Evaluator(
            model=model,
            output_dir=self.output_dir,
            device=device,
        )

        ckpt_basename = os.path.basename(self.checkpoint_path)
        metrics, results = evaluator.evaluate(
            dataloader=eval_loader,
            config=self.config,
            checkpoint_name=ckpt_basename,
            generate_plots=generate_plots,
        )

        return metrics, results
