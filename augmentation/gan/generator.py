"""
Conditional EEG Spectral Generator Network Implementation.
"""

import torch
import torch.nn as nn
from augmentation.gan.base import BaseGenerator


class ConditionalEEGGenerator(BaseGenerator):
    """
    Conditional Generator network mapping latent Gaussian noise vectors z and class labels y
    to synthetic 4D EEG spectral tensors (B, Bands, Channels, Samples).
    """

    def __init__(
        self,
        latent_dim: int = 128,
        num_classes: int = 4,
        num_bands: int = 4,
        num_channels: int = 133,
        num_samples: int = 250,
        hidden_dim: int = 256,
        embed_dim: int = 32,
    ):
        super().__init__(latent_dim=latent_dim, num_classes=num_classes)
        self.num_bands = num_bands
        self.num_channels = num_channels
        self.num_samples = num_samples
        self.target_size = num_bands * num_channels * num_samples

        self.label_emb = nn.Embedding(num_classes, embed_dim)

        self.fc = nn.Sequential(
            nn.Linear(latent_dim + embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 4, self.target_size),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Generate synthetic EEG spectral tensor.

        Args:
            noise: Gaussian noise (B, latent_dim)
            labels: Class labels (B,)

        Returns:
            Generated spectral tensor of shape (B, num_bands, num_channels, num_samples)
        """
        c_emb = self.label_emb(labels)
        x = torch.cat([noise, c_emb], dim=-1)
        flat_out = self.fc(x)
        batch_size = noise.shape[0]
        return flat_out.view(batch_size, self.num_bands, self.num_channels, self.num_samples)
