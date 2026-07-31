"""
Abstract Base Classes for Conditional Generator and Critic Networks.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseGenerator(nn.Module, ABC):
    """Abstract Base Class for EEG Generators."""

    def __init__(self, latent_dim: int = 128, num_classes: int = 4):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Generate synthetic EEG spectral tensor from noise vector z and class labels y.

        Args:
            noise: Tensor of shape (B, latent_dim)
            labels: Tensor of class indices (B,)

        Returns:
            Generated EEG spectral tensor of shape (B, Bands, Channels, Samples)
        """
        pass


class BaseCritic(nn.Module, ABC):
    """Abstract Base Class for EEG Critics (Discriminators)."""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, eeg: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Assess realism score of real vs fake EEG spectral representation.

        Args:
            eeg: Spectral tensor of shape (B, Bands, Channels, Samples)
            labels: Tensor of class indices (B,)

        Returns:
            Scalar score tensor of shape (B, 1)
        """
        pass
