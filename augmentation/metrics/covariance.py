"""
Channel Covariance Matrix Distance Metric Implementation.
"""

import torch
from augmentation.metrics.base import GANMetric


class CovarianceDistance(GANMetric):
    """Computes Frobenius norm distance between real and fake channel covariance matrices."""

    def compute(self, real_eeg: torch.Tensor, fake_eeg: torch.Tensor) -> float:
        """Compute Frobenius distance between covariance matrices."""
        # Reshape to (N, Channels, Bands * Samples)
        B_real, bands, channels, samples = real_eeg.shape
        real_flat = real_eeg.permute(0, 2, 1, 3).reshape(B_real, channels, -1)

        B_fake = fake_eeg.shape[0]
        fake_flat = fake_eeg.permute(0, 2, 1, 3).reshape(B_fake, channels, -1)

        # Average covariance matrix (Channels, Channels)
        cov_real = torch.mean(torch.bmm(real_flat, real_flat.transpose(1, 2)), dim=0) / real_flat.shape[-1]
        cov_fake = torch.mean(torch.bmm(fake_flat, fake_flat.transpose(1, 2)), dim=0) / fake_flat.shape[-1]

        dist = torch.norm(cov_real - cov_fake, p="fro").item()
        return float(dist)
