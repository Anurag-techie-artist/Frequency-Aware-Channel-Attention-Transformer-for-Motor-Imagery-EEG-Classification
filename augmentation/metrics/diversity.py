"""
Synthetic Data Diversity Metric Implementation.
"""

import torch
from augmentation.metrics.base import GANMetric


class DiversityScore(GANMetric):
    """Computes average pairwise distance between generated synthetic EEG samples."""

    def compute(self, real_eeg: torch.Tensor, fake_eeg: torch.Tensor) -> float:
        """Compute mean pairwise L2 distance across synthetic samples."""
        batch_size = fake_eeg.shape[0]
        if batch_size < 2:
            return 0.0

        flat_fake = fake_eeg.view(batch_size, -1)
        pdist = torch.cdist(flat_fake, flat_fake, p=2)
        mean_div = pdist.sum() / (batch_size * (batch_size - 1))
        return float(mean_div.item())
