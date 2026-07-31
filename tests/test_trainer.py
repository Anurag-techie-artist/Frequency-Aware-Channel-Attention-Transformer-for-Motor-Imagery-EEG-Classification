"""
Unit Tests for Trainer (Phase 7).

Tests 1-epoch training and validation loops on synthetic DataLoader.
"""

import os
import sys
import tempfile
import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.trainer import Trainer
from models.eeg_motor_imagery_model import EEGMotorImageryModel
from datasets.builder import create_synthetic_dataset


class TestTrainer(unittest.TestCase):
    """Test suite for Trainer module."""

    def test_train_and_validate_epoch(self):
        """Test train_epoch and validate_epoch execution."""
        config = {
            "model": {
                "transformer": {
                    "d_model": 32,
                    "nhead": 2,
                    "num_layers": 1,
                    "dim_feedforward": 64,
                },
                "classifier": {
                    "hidden_dim": 64,
                },
            },
            "training": {
                "epochs": 1,
                "batch_size": 8,
                "mixed_precision": False,
                "device": "cpu",
            },
            "checkpoint": {
                "save_dir": tempfile.gettempdir(),
                "save_best": False,
            },
        }

        model = EEGMotorImageryModel.from_config(config)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        train_ds = create_synthetic_dataset(num_samples=16)
        val_ds = create_synthetic_dataset(num_samples=8)

        train_loader = DataLoader(train_ds, batch_size=8)
        val_loader = DataLoader(val_ds, batch_size=8)

        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            config=config,
            device=torch.device("cpu"),
        )

        train_metrics = trainer.train_epoch(train_loader)
        self.assertIn("loss", train_metrics)
        self.assertIn("accuracy", train_metrics)

        val_metrics = trainer.validate_epoch(val_loader)
        self.assertIn("loss", val_metrics)
        self.assertIn("accuracy", val_metrics)


if __name__ == "__main__":
    unittest.main()
