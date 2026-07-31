"""
GAN Framework Package.
"""

from augmentation.gan.base import BaseGenerator, BaseCritic
from augmentation.gan.generator import ConditionalEEGGenerator
from augmentation.gan.critic import ConditionalEEGCritic
from augmentation.gan.gradient_penalty import compute_gradient_penalty
from augmentation.gan.losses import wasserstein_loss_critic, wasserstein_loss_generator
from augmentation.gan.state import GANState
from augmentation.gan.checkpoint import GANCheckpointManager
from augmentation.gan.trainer import GANTrainer

__all__ = [
    "BaseGenerator",
    "BaseCritic",
    "ConditionalEEGGenerator",
    "ConditionalEEGCritic",
    "compute_gradient_penalty",
    "wasserstein_loss_critic",
    "wasserstein_loss_generator",
    "GANState",
    "GANCheckpointManager",
    "GANTrainer",
]
