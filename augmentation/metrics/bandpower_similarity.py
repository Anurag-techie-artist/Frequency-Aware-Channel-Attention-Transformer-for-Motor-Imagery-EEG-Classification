"""
Bandpower Distribution Divergence Similarity Metric Implementation.
"""

import torch
import torch.nn.functional as F
from augmentation.metrics.base import GANMetric


class BandPowerSimilarity(GANMetric):
    """Computes bandpower distribution similarity across spectral bands."""

    def compute(self, real_eeg: torch.Tensor, fake_eeg: torch.Tensor) -> float:
        """Compute bandpower similarity score."""
        # Mean energy per band: (Bands,)
        real_bp = torch.mean(real_eeg ** 2, dim=(0, 2, 3))
        fake_bp = torch.mean(fake_eeg ** 2, dim=(0, 2, 3))

        real_p = F.softmax(real_bp, dim=-1)
        fake_p = F.softmax(fake_bp, dim=-1)

        sim = F.cosine_similarity(real_p.unsqueeze(0), fake_p.unsqueeze(0), dim=-1).item()
        return float(sim)
