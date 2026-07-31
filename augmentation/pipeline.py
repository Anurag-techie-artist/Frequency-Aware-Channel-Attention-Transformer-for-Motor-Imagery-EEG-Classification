"""
AugmentationPipeline Module for Applying Data Augmentation to DataLoaders.

Merges real and synthetic EEG datasets without mutating original raw inputs.
"""

from typing import Dict, Any, Tuple
import torch
from torch.utils.data import DataLoader, TensorDataset

from augmentation.strategies.base import AugmentationStrategy
from augmentation.dataset import AugmentedTensorDataset


class AugmentationPipeline:
    """Applies augmentation strategy to datasets and constructs augmented DataLoaders."""

    def __init__(self, strategy: AugmentationStrategy):
        self.strategy = strategy

    def augment_dataloader(
        self,
        real_dataloader: DataLoader,
        ratio: float = 0.5,
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> DataLoader:
        """
        Extract real data tensors from dataloader, run augmentation, and construct augmented DataLoader.

        Args:
            real_dataloader: Input real DataLoader
            ratio: Synthetic augmentation ratio
            batch_size: Target batch size
            shuffle: Whether to shuffle DataLoader

        Returns:
            Augmented PyTorch DataLoader
        """
        all_x = []
        all_y = []
        for x_b, y_b in real_dataloader:
            all_x.append(x_b)
            all_y.append(y_b)

        real_x = torch.cat(all_x, dim=0)
        real_y = torch.cat(all_y, dim=0)

        aug_x, aug_y = self.strategy.augment(real_x, real_y, ratio=ratio)
        aug_ds = TensorDataset(aug_x, aug_y)

        return DataLoader(
            aug_ds,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=torch.cuda.is_available(),
        )
