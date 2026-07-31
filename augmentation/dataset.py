"""
SyntheticDataset Abstract Interface and AugmentedTensorDataset Implementation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import torch
from torch.utils.data import TensorDataset, Dataset


class SyntheticDataset(ABC):
    """Abstract interface for all synthetic dataset generators (GAN, Diffusion, VAE)."""

    @abstractmethod
    def get_data(self) -> torch.Tensor:
        """Return generated synthetic EEG data tensor of shape (N, Bands, Channels, Samples)."""
        pass

    @abstractmethod
    def get_labels(self) -> torch.Tensor:
        """Return generated synthetic class label tensor of shape (N,)."""
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Return generator metadata dictionary."""
        pass


class SimpleSyntheticDataset(SyntheticDataset):
    """Concrete implementation wrapping synthetic tensor data and labels."""

    def __init__(self, data: torch.Tensor, labels: torch.Tensor, metadata: Dict[str, Any] = None):
        self.data = data
        self.labels = labels
        self.metadata = metadata if metadata is not None else {}

    def get_data(self) -> torch.Tensor:
        return self.data

    def get_labels(self) -> torch.Tensor:
        return self.labels

    def get_metadata(self) -> Dict[str, Any]:
        return self.metadata


class AugmentedTensorDataset(Dataset):
    """Combines original dataset tensors with synthetic EEG tensors without mutating raw inputs."""

    def __init__(self, real_x: torch.Tensor, real_y: torch.Tensor, synthetic_x: torch.Tensor, synthetic_y: torch.Tensor):
        self.x = torch.cat([real_x, synthetic_x], dim=0)
        self.y = torch.cat([real_y, synthetic_y], dim=0)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]
