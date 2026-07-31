"""
Conditional EEG Spectral Critic (Discriminator) Implementation.
"""

import torch
import torch.nn as nn
from augmentation.gan.base import BaseCritic


class ConditionalEEGCritic(BaseCritic):
    """
    Conditional Critic network assessing realism score of real vs fake EEG spectral representations.
    Uses LayerNorm to satisfy WGAN-GP 1-Lipschitz continuity constraints.
    """

    def __init__(
        self,
        num_classes: int = 4,
        num_bands: int = 4,
        num_channels: int = 133,
        num_samples: int = 250,
        hidden_dim: int = 256,
        embed_dim: int = 32,
    ):
        super().__init__(num_classes=num_classes)
        self.input_size = num_bands * num_channels * num_samples
        self.label_emb = nn.Embedding(num_classes, embed_dim)

        self.net = nn.Sequential(
            nn.Linear(self.input_size + embed_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, eeg: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute Critic score for input EEG spectral tensor and class label.

        Args:
            eeg: Tensor of shape (B, Bands, Channels, Samples)
            labels: Class label indices (B,)

        Returns:
            Scalar score tensor of shape (B, 1)
        """
        batch_size = eeg.shape[0]
        flat_eeg = eeg.view(batch_size, -1)
        c_emb = self.label_emb(labels)
        x = torch.cat([flat_eeg, c_emb], dim=-1)
        score = self.net(x)
        return score
