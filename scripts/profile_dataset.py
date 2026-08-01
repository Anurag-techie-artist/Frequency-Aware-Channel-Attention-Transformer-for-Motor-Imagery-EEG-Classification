"""
Phase 0 Dataset & DataLoader Bottleneck Profiler Script (v0.11.1).

Measures complete data loading pipeline:
- Dataset Initialization Latency
- First Batch Latency
- Full Pipeline DataLoader Wait Time (Time waiting for next batch)
- Microsecond timing breakdown of __getitem__() (binary search mapping, LRU lookup, slicing, normalization)
- torch.load() invocation count and Cache Hit Ratio
- Memory usage (RAM MB)
- Concurrency benchmark matrix for num_workers = [0, 2, 4, 8]
- Evaluates Phase 0 Decision Gate (Hit Ratio >= 95% threshold)

Usage:
    python scripts/profile_dataset.py [--config configs/train.yaml] [--num_batches 50]
"""

import os
import sys
import time
import logging
import argparse
from typing import Dict, Any, Tuple

import torch
from torch.utils.data import DataLoader

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.config_loader import load_master_config
from datasets.builder import build_dataloaders
from datasets.cache import CACHE_VERSION
from datasets.dataset import HGDDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("profile_dataset")


def parse_args():
    parser = argparse.ArgumentParser(description="Profile HGD Dataset & DataLoader Performance (v0.11.1)")
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
        help="Number of DataLoader batches to benchmark throughput",
    )
    parser.add_argument(
        "--num_micro_samples",
        type=int,
        default=500,
        help="Number of __getitem__ sample calls for microsecond timing breakdown",
    )
    return parser.parse_args()


def get_process_memory_mb() -> float:
    """Get current process RSS memory in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 2)
    except Exception:
        return 0.0


def profile_getitem_microseconds(dataset: HGDDataset, num_samples: int = 500) -> Dict[str, float]:
    """Measure granular timing breakdown of __getitem__() internal operations."""
    if len(dataset) == 0:
        return {}

    indices = [int(i * (len(dataset) - 1) / max(1, num_samples - 1)) for i in range(min(num_samples, len(dataset)))]

    t_mapping_total = 0.0
    t_lru_total = 0.0
    t_slice_total = 0.0
    t_norm_total = 0.0
    t_overall_total = 0.0

    # Ensure dataset cache has reset stats for profiling clean run
    dataset.reset_cache_stats()

    for idx in indices:
        t_start = time.perf_counter()

        # 1. Binary Search Mapping
        from bisect import bisect_right
        t_m0 = time.perf_counter()
        f_idx = bisect_right(dataset._start_indices, idx) - 1
        file_entry = dataset._file_entries[f_idx]
        local_window_idx = idx - file_entry["start_index"]
        t_m1 = time.perf_counter()
        t_mapping_total += (t_m1 - t_m0)

        # 2. LRU Cache Retrieval
        cache_path = os.path.join(dataset.cache_dir, file_entry["cache"])
        t_lru0 = time.perf_counter()
        data = dataset._lru_cache.get(cache_path)
        t_lru1 = time.perf_counter()
        t_lru_total += (t_lru1 - t_lru0)

        # 3. Micro breakdown of Window Slicing vs Normalization
        if "trials" in data:
            trial_start_indices = file_entry["trial_start_indices"]
            trial_idx = bisect_right(trial_start_indices, local_window_idx) - 1
            local_sample_idx = local_window_idx - trial_start_indices[trial_idx]

            window_size = dataset.metadata.get("window_size", 250)
            stride = dataset.metadata.get("stride", 50)
            start_sample = local_sample_idx * stride
            trial_tensor = data["trials"][trial_idx]

            # Slice timing
            t_sl0 = time.perf_counter()
            end_sample = start_sample + window_size
            window_raw = trial_tensor[..., start_sample:end_sample].clone()
            t_sl1 = time.perf_counter()
            t_slice_total += (t_sl1 - t_sl0)

            # Normalization timing
            t_n0 = time.perf_counter()
            mean = torch.mean(window_raw, dim=-1, keepdim=True)
            std = torch.std(window_raw, dim=-1, keepdim=True, correction=0)
            window_norm = (window_raw - mean) / (std + 1e-6)
            t_n1 = time.perf_counter()
            t_norm_total += (t_n1 - t_n0)

        t_end = time.perf_counter()
        t_overall_total += (t_end - t_start)

    n = len(indices)
    return {
        "avg_getitem_ms": (t_overall_total / n) * 1000.0,
        "avg_binary_search_ms": (t_mapping_total / n) * 1000.0,
        "avg_lru_get_ms": (t_lru_total / n) * 1000.0,
        "avg_slice_ms": (t_slice_total / n) * 1000.0,
        "avg_norm_ms": (t_norm_total / n) * 1000.0,
    }


def benchmark_dataloader(loader: DataLoader, max_batches: int = 50) -> Dict[str, Any]:
    """Measure DataLoader wait times, first batch latency, and sample throughput."""
    t_start = time.perf_counter()
    batch_iter = iter(loader)
    
    t0_first = time.perf_counter()
    first_x, first_y = next(batch_iter)
    t1_first = time.perf_counter()
    first_batch_latency = (t1_first - t0_first)

    n_batches = min(max_batches, len(loader))
    total_samples = first_x.size(0)

    wait_times = []

    for _ in range(n_batches - 1):
        t_wait_start = time.perf_counter()
        x, y = next(batch_iter)
        t_wait_end = time.perf_counter()

        wait_times.append(t_wait_end - t_wait_start)
        total_samples += x.size(0)

    t_end = time.perf_counter()
    total_time = t_end - t_start

    avg_wait_ms = (sum(wait_times) / len(wait_times)) * 1000.0 if wait_times else 0.0
    throughput = total_samples / total_time if total_time > 0 else 0.0

    return {
        "first_batch_latency_s": first_batch_latency,
        "avg_batch_wait_ms": avg_wait_ms,
        "total_samples": total_samples,
        "total_time_s": total_time,
        "throughput_samples_per_sec": throughput,
        "batch_size": first_x.size(0),
    }


def main():
    args = parse_args()
    print("=" * 68)
    print("  Phase 0 Dataset & DataLoader Bottleneck Profiler (v0.11.1)")
    print("=" * 68)

    config = load_master_config(train_cfg_path=args.config, project_root=PROJECT_ROOT)
    
    # 1. Dataset Init Latency
    t0_init = time.perf_counter()
    train_loader, val_loader, _ = build_dataloaders(config, project_root=PROJECT_ROOT)
    t1_init = time.perf_counter()
    init_latency_s = t1_init - t0_init

    # Extract underlying dataset for micro-profiling
    raw_dataset = train_loader.dataset
    if hasattr(raw_dataset, "dataset"):  # Subset from random_split
        dataset_obj = raw_dataset.dataset
    else:
        dataset_obj = raw_dataset

    # 2. Microsecond Breakdown of __getitem__
    micro_stats = profile_getitem_microseconds(dataset_obj, num_samples=args.num_micro_samples)

    # 3. Single Config DataLoader Profiling
    dl_stats = benchmark_dataloader(train_loader, max_batches=args.num_batches)

    # 4. Cache Instrumentation
    cache_stats = dataset_obj.get_cache_stats() if hasattr(dataset_obj, "get_cache_stats") else {}

    # 5. Worker Concurrency Matrix Sweep (num_workers = 0, 2, 4, 8)
    concurrency_results = []
    print("\nExecuting Worker Concurrency Matrix Benchmark (num_workers = 0, 2, 4, 8)...")
    for workers in [0, 2, 4, 8]:
        cfg_copy = dict(config)
        cfg_copy.setdefault("training", {})["num_workers"] = workers
        try:
            tr_ld, _, _ = build_dataloaders(cfg_copy, project_root=PROJECT_ROOT)
            bench = benchmark_dataloader(tr_ld, max_batches=min(20, len(tr_ld)))
            concurrency_results.append({
                "workers": workers,
                "first_batch_s": bench["first_batch_latency_s"],
                "wait_ms": bench["avg_batch_wait_ms"],
                "samples_per_sec": bench["throughput_samples_per_sec"],
            })
        except Exception as e:
            logger.warning(f"Failed benchmark for num_workers={workers}: {e}")

    # Output Structured Benchmark Summary
    hit_ratio_pct = cache_stats.get("hit_ratio_pct", 0.0)
    load_count = cache_stats.get("load_count", 0)
    evictions = cache_stats.get("evictions", 0)
    hits = cache_stats.get("hits", 0)
    misses = cache_stats.get("misses", 0)
    peak_ram_mb = cache_stats.get("peak_ram_mb", 0.0)
    process_ram_mb = get_process_memory_mb()

    print("\n" + "=" * 68)
    print("  Complete Input Pipeline Bottleneck Breakdown")
    print("-" * 68)
    print(f"Dataset Init Latency         : {init_latency_s * 1000.0:.2f} ms")
    print(f"First Batch Latency          : {dl_stats['first_batch_latency_s']:.3f} s")
    print(f"DataLoader Batch Wait Time   : {dl_stats['avg_batch_wait_ms']:.2f} ms/batch")
    print(f"DataLoader Sample Throughput  : {dl_stats['throughput_samples_per_sec']:.1f} samples/sec")
    print(f"Process Memory (RSS)         : {process_ram_mb:.1f} MB")
    print("-" * 68)
    print("  __getitem__() Microsecond Latency Breakdown:")
    print(f"  - Overall __getitem__()      : {micro_stats.get('avg_getitem_ms', 0):.3f} ms")
    print(f"  - Binary Search Mapping      : {micro_stats.get('avg_binary_search_ms', 0):.3f} ms")
    print(f"  - LRU Cache Retrieval        : {micro_stats.get('avg_lru_get_ms', 0):.3f} ms")
    print(f"  - Window Tensor Slicing      : {micro_stats.get('avg_slice_ms', 0):.3f} ms")
    print(f"  - Z-score Normalization      : {micro_stats.get('avg_norm_ms', 0):.3f} ms")
    print("-" * 68)
    print("  FileLRUCache Statistics:")
    print(f"  - Cache Version              : {CACHE_VERSION}")
    print(f"  - Cache Hit Ratio            : {hit_ratio_pct:.2f}% ({hits} hits / {misses} misses)")
    print(f"  - torch.load() Count         : {load_count}")
    print(f"  - File Evictions             : {evictions}")
    print(f"  - LRU Memory Footprint       : {peak_ram_mb:.1f} MB")
    print("=" * 68)

    print("\n" + "=" * 68)
    print("  Worker Concurrency Benchmark Matrix (fixed batch_size=32)")
    print("-" * 68)
    print(f"{'num_workers':<12} | {'First Batch (s)':<16} | {'Wait Time (ms)':<16} | {'Samples/sec':<14}")
    print("-" * 68)
    for res in concurrency_results:
        print(f"{res['workers']:<12} | {res['first_batch_s']:<16.3f} | {res['wait_ms']:<16.2f} | {res['samples_per_sec']:<14.1f}")
    print("=" * 68)

    # Decision Gate Assessment
    print("\n" + "=" * 68)
    print("  Phase 0 Decision Gate Evaluation")
    print("-" * 68)
    if hit_ratio_pct < 95.0:
        print(f"  [MISS/THRASH DETECTED] Cache Hit Ratio is {hit_ratio_pct:.2f}% (< 95.0%).")
        print("  -> DECISION: Proceed to Phase 1A (Cache RAM Budget & Memory Optimization).")
    else:
        print(f"  [PASS] Cache Hit Ratio is {hit_ratio_pct:.2f}% (>= 95.0%).")
        print("  -> DECISION: Skip Phase 1A cache modifications entirely.")
        print("  -> DECISION: Proceed to Phase 1B (DataLoader Concurrency Tuning: num_workers, pin_memory).")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
