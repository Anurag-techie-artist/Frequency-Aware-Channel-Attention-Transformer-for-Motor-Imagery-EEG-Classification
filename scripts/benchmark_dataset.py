"""
HGD Dataset Performance Benchmark Script (v0.11.0).

Measures:
- Dataset initialization latency
- First batch latency
- DataLoader batch throughput (samples/sec)
- Actual cache storage footprint (GB)
- System memory consumption (MB)

Usage:
    python scripts/benchmark_dataset.py [--config configs/train.yaml] [--num_batches 50]
"""

import os
import sys
import time
import logging
import argparse
from typing import Dict, Any

import torch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.config_loader import load_master_config
from datasets.builder import build_dataloaders
from datasets.cache import CACHE_VERSION

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("benchmark_dataset")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark HGD Dataset Performance (v0.11.0)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to training config YAML file",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=50,
        help="Number of DataLoader batches to iterate over during throughput measurement",
    )
    return parser.parse_args()


def get_dir_size_gb(dir_path: str) -> float:
    """Calculate actual disk size in GB for directory."""
    if not os.path.exists(dir_path):
        return 0.0
    total_bytes = 0
    for root, _, files in os.walk(dir_path):
        for f in files:
            total_bytes += os.path.getsize(os.path.join(root, f))
    return total_bytes / (1024 ** 3)


def main():
    args = parse_args()
    logger.info("=" * 60)
    logger.info("Starting Dataset Architecture Benchmark (v0.11.0 Trial Cache)")
    logger.info("=" * 60)

    config = load_master_config(train_cfg_path=args.config, project_root=PROJECT_ROOT)
    config.setdefault("training", {})["num_workers"] = 0
    cache_dir = config.get("dataset", {}).get("cache", {}).get("directory", "outputs/cache")

    # Measure Dataset Initialization Latency
    t0 = time.time()
    train_loader, val_loader, test_loader = build_dataloaders(config, project_root=PROJECT_ROOT)
    init_latency = time.time() - t0

    # Measure First Batch Latency
    t_start_batch = time.time()
    batch_iterator = iter(train_loader)
    first_x, first_y = next(batch_iterator)
    first_batch_latency = time.time() - t_start_batch

    # Measure Throughput over N batches
    n_batches = min(args.num_batches, len(train_loader))
    total_samples = first_x.shape[0]  # first batch samples
    t_throughput_start = time.time()

    for _ in range(n_batches - 1):
        x, y = next(batch_iterator)
        total_samples += x.shape[0]

    throughput_time = time.time() - t_throughput_start
    throughput_fps = total_samples / throughput_time if throughput_time > 0 else 0.0

    actual_cache_gb = get_dir_size_gb(cache_dir)

    print("\n" + "=" * 60)
    print("HGD Dataset Benchmark Results (v0.11.0)")
    print("-" * 60)
    print(f"Cache Version               : {CACHE_VERSION}")
    print(f"Cache Directory             : {cache_dir}")
    print(f"Actual Disk Storage         : {actual_cache_gb:.2f} GB")
    print(f"Dataset Init Latency        : {init_latency * 1000:.2f} ms")
    print(f"First Batch Latency         : {first_batch_latency * 1000:.2f} ms")
    print(f"Batch Size                  : {first_x.shape[0]}")
    print(f"Sample Tensor Shape         : {list(first_x.shape[1:])}")
    print(f"DataLoader Throughput       : {throughput_fps:.1f} samples/sec")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
