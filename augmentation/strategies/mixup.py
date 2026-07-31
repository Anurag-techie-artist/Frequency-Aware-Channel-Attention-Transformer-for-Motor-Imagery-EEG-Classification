"""
MixUp EEG Data Augmentation Strategy Implementation.
"""

from typing import Dict, Any, Tuple
import torch
from torch.utils.data import DataLoader
from augmentation.strategies.base import AugmentationStrategy
from augmentation.dataset import SyntheticDataset, SimpleSyntheticDataset


class MixUpStrategy(AugmentationStrategy):
    """MixUp convex interpolation strategy for EEG spectral tensors."""

    def __init__(self, alpha: float = 0.2, seed: int = 42):
        super().__init__(seed=seed)
        self.alpha = alpha

    def fit(self, dataloader: DataLoader, config: Dict[str, Any]):
        pass

    def generate(self, num_samples: int, num_classes: int = 4) -> SyntheticDataset:
        empty_x = torch.empty(0, 4, 133, 250)
        empty_y = torch.empty(0, dtype=torch.long)
        return SimpleSyntheticDataset(empty_x, empty_y, {"strategy": "mixup"})

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

        gen = torch.Generator().manual_seed(self.seed)
        idx1 = torch.randint(0, num_real, (num_synth,), generator=gen)
        idx2 = torch.randint(0, num_real, (num_synth,), generator=gen)

        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((num_synth, 1, 1, 1))

        mix_x = lam * real_x[idx1] + (1 - lam) * real_x[idx2]
        mix_y = real_y[idx1]  # Dominant label assignment

        aug_x = torch.cat([real_x, mix_x], dim=0)
        aug_y = torch.cat([real_y, mix_y], dim=0)
        return aug_x, aug_y
