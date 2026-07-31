"""
Unit Tests for Training Resumption (Phase 7).

Tests that training 2 epochs, saving checkpoint, and resuming from latest.pt
continues cleanly at epoch 3.
"""

import os
import sys
import tempfile
import unittest
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.trainer import Trainer
from models.eeg_motor_imagery_model import EEGMotorImageryModel
from datasets.builder import create_synthetic_dataset
from torch.utils.data import DataLoader


class TestResumeTraining(unittest.TestCase):
    """Test suite for training resumption parity."""

    def test_resume_training_epoch_advancement(self):
        """Test resuming from checkpoint starts at next epoch."""
        with tempfile.TemporaryDirectory() as tmp_dir:
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
                    "synthetic_data": True,
                    "epochs": 2,
                    "batch_size": 8,
                    "mixed_precision": False,
                    "device": "cpu",
                },
                "checkpoint": {
                    "save_dir": tmp_dir,
                    "save_best": True,
                    "monitor": "val_accuracy",
                },
            }

            model = EEGMotorImageryModel.from_config(config)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

            train_ds = create_synthetic_dataset(num_samples=16)
            val_ds = create_synthetic_dataset(num_samples=8)

            train_loader = DataLoader(train_ds, batch_size=8)
            val_loader = DataLoader(val_ds, batch_size=8)

            trainer1 = Trainer(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                config=config,
                device=torch.device("cpu"),
            )

            # Train 2 epochs
            state1 = trainer1.fit(train_loader, val_loader)
            self.assertEqual(state1.epoch, 2)

            ckpt_path = os.path.join(tmp_dir, "latest.pt")
            self.assertTrue(os.path.exists(ckpt_path))

            # Resume training for 2 additional epochs (total 4 epochs)
            config["training"]["epochs"] = 4
            model2 = EEGMotorImageryModel.from_config(config)
            optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)

            trainer2 = Trainer(
                model=model2,
                criterion=criterion,
                optimizer=optimizer2,
                config=config,
                device=torch.device("cpu"),
            )

            state2 = trainer2.fit(train_loader, val_loader, resume_path=ckpt_path)
            self.assertEqual(state2.epoch, 4)


if __name__ == "__main__":
    unittest.main()
