"""
Offline Dataset Cache Precomputation Script.

Scans the High Gamma Dataset (HGD) train and test directories, processes EDF files,
and builds/updates per-EDF `.pt` files and `metadata.json` in `outputs/cache/`.
Phase 10 Patch v0.10.4: Production-Grade Lazy HGD Dataset Layer.

Usage:
    python scripts/precompute_dataset.py [--config configs/train.yaml]
"""

import os
import sys
import time
import logging
import argparse

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.config_loader import load_master_config
from datasets.path import get_dataset_root, get_train_directory, get_test_directory, validate_dataset
from datasets.loader import discover_edf_files
from datasets.pipeline import EEGPreprocessingPipeline, PreprocessingConfig
from datasets.cache import CacheManager, compute_config_hash

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("precompute_dataset")


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute HGD Dataset Cache")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to training config YAML file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info(f"Starting dataset cache precomputation with config: {args.config}")

    config = load_master_config(train_cfg_path=args.config, project_root=PROJECT_ROOT)
    dataset_cfg = config.get("dataset", {})
    cache_cfg = dataset_cfg.get("cache", {})

    cache_dir = cache_cfg.get("directory", "outputs/cache")
    representation = dataset_cfg.get(
        "representation",
        "frequency" if config.get("frequency", {}).get("enabled", False) else "time",
    )

    validate_dataset(project_root=PROJECT_ROOT)
    dataset_root = get_dataset_root(project_root=PROJECT_ROOT)
    train_dir = os.path.join(dataset_root, get_train_directory(project_root=PROJECT_ROOT))
    test_dir = os.path.join(dataset_root, get_test_directory(project_root=PROJECT_ROOT))

    train_files = discover_edf_files(train_dir)
    test_files = discover_edf_files(test_dir)

    if not train_files:
        raise FileNotFoundError(f"No training EDF files found in: {train_dir}")
    if not test_files:
        raise FileNotFoundError(f"No testing EDF files found in: {test_dir}")

    all_files = train_files + test_files
    logger.info(f"Discovered {len(train_files)} train files and {len(test_files)} test files.")

    prep_config = PreprocessingConfig.from_dict(config)
    pipeline = EEGPreprocessingPipeline(config=prep_config)
    config_hash = compute_config_hash(pipeline.config, representation)

    cache_manager = CacheManager(cache_dir=cache_dir)

    t0 = time.time()
    meta = cache_manager.build_cache(
        file_paths=all_files,
        pipeline=pipeline,
        representation=representation,
        config_hash=config_hash,
        build_if_missing=True,
    )
    elapsed = time.time() - t0

    print("\n" + "=" * 50)
    print("Cache created")
    print(f"{len(train_files)} train tensors")
    print(f"{len(test_files)} test tensors")
    print(f"Total Samples: {meta.get('total_samples', 0)}")
    print(f"Execution Time: {elapsed:.2f} seconds")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
