"""
Dataset & DataLoader Factory for EEGMotorImageryModel Training.

Constructs PyTorch DataLoaders for training, validation, and test sets using real
High-Gamma Dataset (HGD) preprocessing pipeline or synthetic fallback.
Phase 10 Patch v0.10.4: Production-Grade Lazy HGD Dataset Layer.
"""

import os
import logging
from typing import Dict, Any, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

from datasets.path import (
    get_dataset_root,
    get_train_directory,
    get_test_directory,
    validate_dataset,
)
from datasets.loader import discover_edf_files
from datasets.pipeline import EEGPreprocessingPipeline, PreprocessingConfig
from datasets.dataset import HGDDataset
from datasets.cache import CACHE_VERSION

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
    model_cfg = config.get("model", {})
    dataset_cfg = config.get("dataset", {})

    batch_size = train_cfg.get("batch_size", 32)
    num_workers = train_cfg.get("num_workers", 0)
    seed = train_cfg.get("seed", 42)
    synthetic_data = train_cfg.get("synthetic_data", config.get("synthetic_data", False))
    val_split_ratio = float(train_cfg.get("validation_split", 0.2))
    split_strategy = train_cfg.get("split_strategy", "random")

    representation = dataset_cfg.get(
        "representation",
        config.get(
            "representation",
            "frequency" if config.get("frequency", {}).get("enabled", False) else "time",
        ),
    )
    cache_cfg = dataset_cfg.get("cache", {})

    if synthetic_data:
        logger.info("Building synthetic dataset DataLoaders (synthetic_data=True)...")
        dataset_root = "Synthetic (In-Memory)"
        train_files_count = 0
        test_files_count = 0
        cache_status = "N/A (Synthetic)"
        cache_dir = "N/A"
        max_open_cache = 0

        train_ds = create_synthetic_dataset(
            num_samples=128,
            num_bands=model_cfg.get("num_bands", 4),
            num_channels=model_cfg.get("num_channels", 133),
            num_samples_per_window=model_cfg.get("num_samples", 250),
            num_classes=4,
            seed=seed,
        )
        val_ds = create_synthetic_dataset(
            num_samples=32,
            num_bands=model_cfg.get("num_bands", 4),
            num_channels=model_cfg.get("num_channels", 133),
            num_samples_per_window=model_cfg.get("num_samples", 250),
            num_classes=4,
            seed=seed + 1,
        )
        test_ds = create_synthetic_dataset(
            num_samples=32,
            num_bands=model_cfg.get("num_bands", 4),
            num_channels=model_cfg.get("num_channels", 133),
            num_samples_per_window=model_cfg.get("num_samples", 250),
            num_classes=4,
            seed=seed + 2,
        )
    else:
        validate_dataset(project_root=project_root)
        dataset_root = get_dataset_root(project_root=project_root)
        train_dir_name = get_train_directory(project_root=project_root)
        test_dir_name = get_test_directory(project_root=project_root)

        train_dir = os.path.join(dataset_root, train_dir_name)
        test_dir = os.path.join(dataset_root, test_dir_name)

        train_files = discover_edf_files(train_dir)
        test_files = discover_edf_files(test_dir)

        if not train_files:
            raise FileNotFoundError(
                f"No EDF files found in training directory: '{train_dir}'. "
                f"Please check your dataset path configuration."
            )
        if not test_files:
            raise FileNotFoundError(
                f"No EDF files found in testing directory: '{test_dir}'. "
                f"Please check your dataset path configuration."
            )

        train_files_count = len(train_files)
        test_files_count = len(test_files)

        prep_config = PreprocessingConfig.from_dict(config)
        pipeline = EEGPreprocessingPipeline(config=prep_config)

        full_train_ds = HGDDataset(
            file_paths=train_files,
            pipeline=pipeline,
            representation=representation,
            cache_config=cache_cfg,
        )
        test_ds = HGDDataset(
            file_paths=test_files,
            pipeline=pipeline,
            representation=representation,
            cache_config=cache_cfg,
        )

        cache_status = "HIT" if full_train_ds.metadata else "MISS"
        cache_dir = full_train_ds.cache_dir
        max_open_cache = full_train_ds.max_open_cache_files

        if split_strategy == "random":
            total_len = len(full_train_ds)
            val_len = int(total_len * val_split_ratio)
            train_len = total_len - val_len
            generator = torch.Generator().manual_seed(seed)
            train_ds, val_ds = random_split(full_train_ds, [train_len, val_len], generator=generator)
        else:
            raise ValueError(f"Unsupported split strategy '{split_strategy}'. Supported strategies: ['random']")

    window_size = config.get("window_size", 250)
    stride = config.get("stride", 50)

    summary_msg = f"""
==================================================
Dataset Summary
--------------------------------------------------
Dataset Root         : {dataset_root}
Representation       : {representation}
Window Size          : {window_size}
Stride               : {stride}

Cache Status         : {cache_status}
Cache Version        : {CACHE_VERSION}
Cache Directory      : {cache_dir}
Cached EDF Files     : {train_files_count + test_files_count}
Open Cache Limit     : {max_open_cache}

Training Samples     : {len(train_ds)}
Validation Samples   : {len(val_ds)}
Testing Samples      : {len(test_ds)}

Batch Size           : {batch_size}
Workers              : {num_workers}
==================================================
"""
    logger.info(summary_msg)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
