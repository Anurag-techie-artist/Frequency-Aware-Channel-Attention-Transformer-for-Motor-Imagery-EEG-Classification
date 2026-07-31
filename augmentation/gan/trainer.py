"""
GANTrainer Module for Conditional WGAN-GP Training Loop Execution.
"""

import os
import time
import logging
from typing import Dict, Any, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from augmentation.gan.generator import ConditionalEEGGenerator
from augmentation.gan.critic import ConditionalEEGCritic
from augmentation.gan.gradient_penalty import compute_gradient_penalty
from augmentation.gan.losses import wasserstein_loss_critic, wasserstein_loss_generator
from augmentation.gan.checkpoint import GANCheckpointManager
from training.device import get_device

logger = logging.getLogger(__name__)


class GANTrainer:
    """Manages Conditional WGAN-GP optimization, 5:1 Critic updates, gradient penalty, and checkpoints."""

    def __init__(
        self,
        generator: ConditionalEEGGenerator,
        critic: ConditionalEEGCritic,
        config: Dict[str, Any],
        device: torch.device = None,
    ):
        self.config = config
        gan_cfg = config.get("gan", {})
        out_cfg = config.get("output", {})

        self.device = device if device else get_device(gan_cfg.get("device", "auto"))
        self.generator = generator.to(self.device)
        self.critic = critic.to(self.device)

        self.epochs = int(gan_cfg.get("epochs", 200))
        self.critic_steps = int(gan_cfg.get("critic_steps", 5))
        self.gp_lambda = float(gan_cfg.get("gradient_penalty_lambda", 10.0))
        self.lr = float(gan_cfg.get("learning_rate", 1e-4))
        self.beta1 = float(gan_cfg.get("beta1", 0.5))
        self.beta2 = float(gan_cfg.get("beta2", 0.9))

        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2)
        )
        self.c_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.lr, betas=(self.beta1, self.beta2)
        )

        out_dir = out_cfg.get("output_dir", "outputs/augmentation")
        gan_dir = os.path.join(out_dir, "gan")
        self.checkpoint_manager = GANCheckpointManager(save_dir=gan_dir)
        self.history: List[Dict[str, Any]] = []

    def fit(self, dataloader: DataLoader) -> List[Dict[str, Any]]:
        """
        Execute WGAN-GP training over specified number of epochs.

        Args:
            dataloader: DataLoader yielding (eeg_tensor, labels)

        Returns:
            List of epoch metric dictionary records
        """
        logger.info(f"Starting WGAN-GP Training for {self.epochs} epochs on device: {self.device}")
        global_step = 0

        for epoch in range(1, self.epochs + 1):
            epoch_c_loss = 0.0
            epoch_g_loss = 0.0
            epoch_w_dist = 0.0
            n_c_steps = 0
            n_g_steps = 0

            for batch_idx, (real_eeg, labels) in enumerate(dataloader):
                real_eeg = real_eeg.to(self.device)
                labels = labels.to(self.device)
                batch_size = real_eeg.size(0)

                # ==========================================
                # 1. Update Critic (critic_steps times)
                # ==========================================
                self.c_optimizer.zero_grad()

                noise = torch.randn(batch_size, self.generator.latent_dim, device=self.device)
                fake_eeg = self.generator(noise, labels).detach()

                real_scores = self.critic(real_eeg, labels)
                fake_scores = self.critic(fake_eeg, labels)

                gp = compute_gradient_penalty(
                    self.critic, real_eeg, fake_eeg, labels, self.device
                )
                c_loss = wasserstein_loss_critic(
                    real_scores, fake_scores, gradient_penalty=gp, gp_lambda=self.gp_lambda
                )

                c_loss.backward()
                self.c_optimizer.step()

                w_dist = (real_scores.mean() - fake_scores.mean()).item()
                epoch_c_loss += c_loss.item()
                epoch_w_dist += w_dist
                n_c_steps += 1

                # ==========================================
                # 2. Update Generator (every critic_steps)
                # ==========================================
                if (batch_idx + 1) % self.critic_steps == 0:
                    self.g_optimizer.zero_grad()
                    gen_noise = torch.randn(batch_size, self.generator.latent_dim, device=self.device)
                    gen_fake_eeg = self.generator(gen_noise, labels)
                    gen_scores = self.critic(gen_fake_eeg, labels)

                    g_loss = wasserstein_loss_generator(gen_scores)
                    g_loss.backward()
                    self.g_optimizer.step()

                    epoch_g_loss += g_loss.item()
                    n_g_steps += 1

                global_step += 1

            avg_c_loss = epoch_c_loss / max(n_c_steps, 1)
            avg_g_loss = epoch_g_loss / max(n_g_steps, 1)
            avg_w_dist = epoch_w_dist / max(n_c_steps, 1)

            record = {
                "epoch": epoch,
                "critic_loss": avg_c_loss,
                "generator_loss": avg_g_loss,
                "wasserstein_distance": avg_w_dist,
            }
            self.history.append(record)

            if epoch % 10 == 0 or epoch == self.epochs:
                logger.info(
                    f"GAN Epoch {epoch:03d}/{self.epochs:03d} | Critic Loss: {avg_c_loss:.4f} | "
                    f"Gen Loss: {avg_g_loss:.4f} | W-Dist: {avg_w_dist:.4f}"
                )

        # Save final GAN checkpoint
        self.checkpoint_manager.save_checkpoint(
            generator=self.generator,
            critic=self.critic,
            g_optimizer=self.g_optimizer,
            c_optimizer=self.c_optimizer,
            epoch=self.epochs,
            global_step=global_step,
            history=self.history,
        )
        return self.history
