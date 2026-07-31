"""
Dataset & DataLoader Factory for EEGMotorImageryModel Training.

Constructs PyTorch DataLoaders for training, validation, and test sets.
Supports synthetic dataset fallbacks for standalone framework verification.
"""

import os
import logging
from typing import Dict, Any, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

logger = logging.getLogger(__name__)


def create_synthetic_dataset(
    num_samples: int = 64,
    num_bands: int = 4,
    num_channels: int = 133,
    num_samples_per_window: int = 250,
    num_classes: int = 4,
    seed: int = 42,
) -> TensorDataset:
    """Create a synthetic TensorDataset for framework testing and verification."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(
        num_samples,
        num_bands,
        num_channels,
        num_samples_per_window,
        generator=generator,
    )
    y = torch.randint(0, num_classes, (num_samples,), generator=generator)
    return TensorDataset(x, y)


def build_dataloaders(
    config: Dict[str, Any],
    project_root: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Build train, val, and test DataLoaders from configuration dictionary.

    Args:
        config: Master merged configuration dictionary
        project_root: Optional path to project root directory

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_cfg = config.get("training", {})
    batch_size = train_cfg.get("batch_size", 32)
    num_workers = train_cfg.get("num_workers", 0)
    seed = train_cfg.get("seed", 42)

    # For now, construct synthetic fallback dataset to enable pipeline verification
    # Future datasets (HGD, BCI IV-2a) connect seamlessly via Dataset interface
    train_ds = create_synthetic_dataset(
        num_samples=128, num_classes=4, seed=seed
    )
    val_ds = create_synthetic_dataset(
        num_samples=32, num_classes=4, seed=seed + 1
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, None
