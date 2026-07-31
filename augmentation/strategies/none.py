"""
No-Op Baseline Augmentation Strategy Implementation.
"""

from typing import Dict, Any, Tuple
import torch
from torch.utils.data import DataLoader
from augmentation.strategies.base import AugmentationStrategy
from augmentation.dataset import SyntheticDataset, SimpleSyntheticDataset


class NoAugmentationStrategy(AugmentationStrategy):
    """No-op baseline strategy returning real dataset unaugmented."""

    def fit(self, dataloader: DataLoader, config: Dict[str, Any]):
        pass

    def generate(self, num_samples: int, num_classes: int = 4) -> SyntheticDataset:
        empty_x = torch.empty(0, 4, 133, 250)
        empty_y = torch.empty(0, dtype=torch.long)
        return SimpleSyntheticDataset(empty_x, empty_y, {"strategy": "none"})

    def augment(
        self,
        real_x: torch.Tensor,
        real_y: torch.Tensor,
        ratio: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return real_x, real_y
