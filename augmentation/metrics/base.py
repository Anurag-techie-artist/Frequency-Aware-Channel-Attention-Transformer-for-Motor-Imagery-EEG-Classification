"""
Abstract Base Class for Standardized GAN Metrics.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import torch


class GANMetric(ABC):
    """Abstract Base Class for GAN quality evaluation metrics."""

    @abstractmethod
    def compute(self, real_eeg: torch.Tensor, fake_eeg: torch.Tensor) -> float:
        """
        Compute quality metric comparison score between real and fake EEG tensors.

        Args:
            real_eeg: Tensor of shape (N_real, Bands, Channels, Samples)
            fake_eeg: Tensor of shape (N_fake, Bands, Channels, Samples)

        Returns:
            Scalar metric value
        """
        pass
