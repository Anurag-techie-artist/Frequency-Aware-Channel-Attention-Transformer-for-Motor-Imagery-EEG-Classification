"""
Power Spectral Density (PSD) Similarity Metric Implementation.
"""

import torch
import torch.nn.functional as F
from augmentation.metrics.base import GANMetric


class PSDSimilarity(GANMetric):
    """Computes Cosine Similarity between average Power Spectral Densities of real vs fake EEG."""

    def compute(self, real_eeg: torch.Tensor, fake_eeg: torch.Tensor) -> float:
        """Compute PSD cosine similarity score in [0.0, 1.0]."""
        # Mean spectral power across samples: (Bands, Channels)
        real_psd = torch.mean(real_eeg ** 2, dim=(-1, 0)).flatten()
        fake_psd = torch.mean(fake_eeg ** 2, dim=(-1, 0)).flatten()

        sim = F.cosine_similarity(real_psd.unsqueeze(0), fake_psd.unsqueeze(0), dim=-1).item()
        return float(sim)
