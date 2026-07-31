"""
Abstract Base Class for Augmentation Strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import torch
from torch.utils.data import DataLoader, TensorDataset
from augmentation.dataset import SyntheticDataset


class AugmentationStrategy(ABC):
    """Abstract Base Class for all EEG data augmentation techniques."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    @abstractmethod
    def fit(self, dataloader: DataLoader, config: Dict[str, Any]):
        """Fit augmentation model (e.g. train GAN generator) on real training dataloader."""
        pass

    @abstractmethod
    def generate(self, num_samples: int, num_classes: int = 4) -> SyntheticDataset:
        """Generate synthetic EEG data dataset."""
        pass

    @abstractmethod
    def augment(
        self,
        real_x: torch.Tensor,
        real_y: torch.Tensor,
        ratio: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Augment real dataset with synthetic dataset to achieve specified ratio."""
        pass
