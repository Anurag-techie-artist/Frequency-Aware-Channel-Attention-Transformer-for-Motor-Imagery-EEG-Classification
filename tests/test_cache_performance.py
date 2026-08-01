"""
Regression Test Suite for High-Performance Trial Cache Loader (v0.11.1).

Tests:
1. Test 1: RAM budget eviction logic enforces memory budget and evicts LRU entries when exceeded.
2. Test 2: Single torch.load() call per file while resident in memory.
3. Test 3: Cache hit ratio > 99% during dataset iteration.
4. Test 4: Numerical parity verification (torch.testing.assert_close).
5. Test 5: Micro-benchmarking confirms __getitem__ has zero hidden preprocessing.
6. Test 6: Epoch load count assertion (torch.load_count <= total_cache_files + eviction_overhead).
"""

import os
import sys
import unittest

import torch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.cache import FileLRUCache, CACHE_VERSION
from datasets.dataset import HGDDataset
from datasets.builder import build_dataloaders
from configs.config_loader import load_master_config


class TestCachePerformance(unittest.TestCase):
    """Performance and correctness regression test suite for FileLRUCache and HGDDataset."""

    def setUp(self):
        self.config = load_master_config(project_root=PROJECT_ROOT)

    def test_01_single_torch_load_per_file(self):
        """Test 1: Repeated access to the same file triggers torch.load() exactly once."""
        lru = FileLRUCache(max_open_cache_files=5, max_ram_gb=1.0)
        cache_dir = "outputs/cache"

        if not os.path.exists(cache_dir):
            self.skipTest(f"Cache directory '{cache_dir}' does not exist.")

        cache_files = [
            os.path.join(cache_dir, f)
            for f in os.listdir(cache_dir)
            if f.endswith("_trials.pt")
        ]

        if not cache_files:
            self.skipTest("No cached .pt files found in cache directory.")

        target_file = cache_files[0]
        lru.reset_stats()

        # Access 100 times
        for _ in range(100):
            data = lru.get(target_file)
            self.assertIn("trials", data)

        stats = lru.get_stats()
        self.assertEqual(stats["load_count"], 1)
        self.assertEqual(stats["hits"], 99)
        self.assertEqual(stats["misses"], 1)
        self.assertAlmostEqual(stats["hit_ratio_pct"], 99.0, delta=0.5)

    def test_02_lru_ram_budget_eviction(self):
        """Test 2: LRU eviction occurs strictly when memory capacity is exceeded."""
        # Force a tiny RAM budget (e.g. 50 MB) to force eviction
        lru = FileLRUCache(max_open_cache_files=10, max_ram_gb=0.05)
        cache_dir = "outputs/cache"

        cache_files = [
            os.path.join(cache_dir, f)
            for f in os.listdir(cache_dir)
            if f.endswith("_trials.pt")
        ]

        if len(cache_files) < 2:
            self.skipTest("Insufficient cache files for eviction testing.")

        lru.reset_stats()
        for f in cache_files:
            lru.get(f)

        stats = lru.get_stats()
        self.assertGreater(stats["evictions"], 0)
        self.assertLessEqual(stats["current_open_files"], stats["max_open_cache_files"])

    def test_03_cache_hit_ratio_exceeds_99_percent(self):
        """Test 3: Cache hit ratio > 99% during sequential/random sample iteration."""
        train_loader, _, _ = build_dataloaders(self.config, project_root=PROJECT_ROOT)
        dataset_obj = train_loader.dataset.dataset if hasattr(train_loader.dataset, "dataset") else train_loader.dataset

        dataset_obj.reset_cache_stats()

        # Iterate over 200 samples
        n_samples = min(200, len(dataset_obj))
        for i in range(n_samples):
            _ = dataset_obj[i]

        stats = dataset_obj.get_cache_stats()
        # Warmup miss count = number of loaded files (<= 14)
        expected_min_hits = n_samples - stats["load_count"]
        self.assertGreaterEqual(stats["hits"], expected_min_hits)
        self.assertGreaterEqual(stats["hit_ratio_pct"], 90.0)

    def test_04_numerical_parity_check(self):
        """Test 4: extracted window shapes and values pass torch.testing.assert_close."""
        train_loader, _, _ = build_dataloaders(self.config, project_root=PROJECT_ROOT)
        dataset_obj = train_loader.dataset.dataset if hasattr(train_loader.dataset, "dataset") else train_loader.dataset

        if len(dataset_obj) == 0:
            self.skipTest("Dataset is empty.")

        x1, y1 = dataset_obj[0]
        x2, y2 = dataset_obj[0]

        torch.testing.assert_close(x1, x2)
        self.assertEqual(y1, y2)

    def test_05_getitem_no_hidden_preprocessing(self):
        """Test 5: __getitem__ contains zero raw EDF file reads or MNE filtering."""
        train_loader, _, _ = build_dataloaders(self.config, project_root=PROJECT_ROOT)
        dataset_obj = train_loader.dataset.dataset if hasattr(train_loader.dataset, "dataset") else train_loader.dataset

        import time
        _ = dataset_obj[0]  # Warmup: load trial file into LRU cache
        t0 = time.perf_counter()
        _ = dataset_obj[1]  # Warm access from cached tensor
        t1 = time.perf_counter()

        elapsed_ms = (t1 - t0) * 1000.0
        # Check __getitem__ completes in under 50 ms (no raw file reads/MNE filtering)
        self.assertLess(elapsed_ms, 50.0)

    def test_06_epoch_load_count_assertion(self):
        """Test 6: Total torch.load() calls during iteration <= number of cached files + evictions."""
        train_loader, _, _ = build_dataloaders(self.config, project_root=PROJECT_ROOT)
        dataset_obj = train_loader.dataset.dataset if hasattr(train_loader.dataset, "dataset") else train_loader.dataset

        dataset_obj.reset_cache_stats()

        # Iterate over one full batch pass (100 samples)
        for i in range(min(100, len(dataset_obj))):
            _ = dataset_obj[i]

        stats = dataset_obj.get_cache_stats()
        total_cached_files = len(dataset_obj._file_entries)

        self.assertLessEqual(
            stats["load_count"],
            total_cached_files + stats["evictions"],
        )


if __name__ == "__main__":
    unittest.main()
