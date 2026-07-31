"""
Unit Tests for GANTrainer (Phase 10).
"""

import os
import sys
import unittest
import tempfile
import torch
from torch.utils.data import TensorDataset, DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from augmentation.gan.generator import ConditionalEEGGenerator
from augmentation.gan.critic import ConditionalEEGCritic
from augmentation.gan.trainer import GANTrainer


class TestGANTrainer(unittest.TestCase):
    """Test suite for GANTrainer execution and history logging."""

    def test_gan_trainer_fit(self):
        """Test GANTrainer runs 2 training epochs and records loss history."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            gen = ConditionalEEGGenerator(latent_dim=16, num_classes=4, num_bands=2, num_channels=5, num_samples=20)
            critic = ConditionalEEGCritic(num_classes=4, num_bands=2, num_channels=5, num_samples=20)

            config = {
                "gan": {
                    "epochs": 2,
                    "critic_steps": 1,
                    "latent_dim": 16,
                    "gradient_penalty_lambda": 10.0,
                    "learning_rate": 1e-3,
                    "device": "cpu",
                },
                "output": {"output_dir": tmp_dir},
            }

            dataset = TensorDataset(torch.randn(16, 2, 5, 20), torch.randint(0, 4, (16,)))
            loader = DataLoader(dataset, batch_size=8)

            trainer = GANTrainer(gen, critic, config, device=torch.device("cpu"))
            history = trainer.fit(loader)

            self.assertEqual(len(history), 2)
            self.assertIn("critic_loss", history[0])
            self.assertIn("generator_loss", history[0])


if __name__ == "__main__":
    unittest.main()
