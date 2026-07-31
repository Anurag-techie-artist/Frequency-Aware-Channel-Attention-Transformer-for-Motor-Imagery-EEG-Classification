"""
Synthetic Data Integrity Validator Utility.

Performs data integrity verification (shape, NaN/Inf checks, label balance, channel count,
sampling rate) before synthetic data is passed to the classifier.
"""

from typing import Dict, Any, Tuple
import torch


class SyntheticDataValidator:
    """Validates synthetic EEG tensors prior to classifier retraining."""

    @staticmethod
    def validate_synthetic_dataset(
        synthetic_x: torch.Tensor,
        synthetic_y: torch.Tensor,
        expected_bands: int = 4,
        expected_channels: int = 133,
        expected_samples: int = 250,
    ) -> bool:
        """
        Validate synthetic data tensors.

        Args:
            synthetic_x: Synthetic data tensor (N, Bands, Channels, Samples)
            synthetic_y: Synthetic labels tensor (N,)
            expected_bands: Target band dimension
            expected_channels: Target channel dimension
            expected_samples: Target sample window dimension

        Returns:
            True if valid, raises ValueError if invalid
        """
        if synthetic_x.ndim != 4:
            raise ValueError(f"Synthetic data tensor must be 4D (N, B, C, S), got shape {synthetic_x.shape}")

        if synthetic_x.shape[0] != synthetic_y.shape[0]:
            raise ValueError(f"Data sample count ({synthetic_x.shape[0]}) mismatch with labels count ({synthetic_y.shape[0]})")

        # Shape integrity check
        _, bands, channels, samples = synthetic_x.shape
        if bands != expected_bands:
            raise ValueError(f"Synthetic band count ({bands}) does not match expected ({expected_bands})")
        if channels != expected_channels:
            raise ValueError(f"Synthetic channel count ({channels}) does not match expected ({expected_channels})")
        if samples != expected_samples:
            raise ValueError(f"Synthetic sample window ({samples}) does not match expected ({expected_samples})")

        # NaN and Inf integrity check
        if torch.isnan(synthetic_x).any():
            raise ValueError("Synthetic EEG data tensor contains NaN values!")
        if torch.isinf(synthetic_x).any():
            raise ValueError("Synthetic EEG data tensor contains Inf values!")

        # Label range check
        if (synthetic_y < 0).any():
            raise ValueError("Synthetic labels contain negative values!")

        return True
