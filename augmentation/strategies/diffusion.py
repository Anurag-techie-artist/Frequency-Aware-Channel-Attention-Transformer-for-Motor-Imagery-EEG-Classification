"""
Diffusion Model EEG Synthetic Augmentation Strategy Placeholder with Fallback.
"""

from typing import Dict, Any, Tuple
import torch
from torch.utils.data import DataLoader
from augmentation.strategies.base import AugmentationStrategy
from augmentation.dataset import SyntheticDataset, SimpleSyntheticDataset


class DiffusionStrategy(AugmentationStrategy):
    """Diffusion strategy placeholder for future score-based EEG generation."""

    def __init__(self, seed: int = 42):
        super().__init__(seed=seed)

    def fit(self, dataloader: DataLoader, config: Dict[str, Any]):
        pass

    def generate(self, num_samples: int, num_classes: int = 4) -> SyntheticDataset:
        gen = torch.Generator().manual_seed(self.seed)
        diff_x = torch.randn(num_samples, 4, 133, 250, generator=gen)
        diff_y = torch.randint(0, num_classes, (num_samples,), generator=gen)
        return SimpleSyntheticDataset(diff_x, diff_y, {"strategy": "diffusion"})

    def augment(
        self,
        real_x: torch.Tensor,
        real_y: torch.Tensor,
        ratio: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if ratio <= 0.0:
            return real_x, real_y

        num_real = real_x.shape[0]
        num_synth = int(num_real * ratio)
        if num_synth <= 0:
            return real_x, real_y

        ds = self.generate(num_samples=num_synth)
        aug_x = torch.cat([real_x, ds.get_data()], dim=0)
        aug_y = torch.cat([real_y, ds.get_labels()], dim=0)
        return aug_x, aug_y
