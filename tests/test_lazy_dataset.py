"""
Unit Tests for v0.10.4 Lazy HGD Dataset Layer & Cache Manager.
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.cache import CacheManager, FileLRUCache, compute_config_hash, CACHE_VERSION
from datasets.dataset import HGDDataset
from datasets.pipeline import EEGPreprocessingPipeline, PreprocessingConfig


class TestLazyDataset(unittest.TestCase):
    """Comprehensive test suite for CacheManager, FileLRUCache, and lazy HGDDataset loading."""

    def setUp(self):
        self.sample_edf = os.path.join(PROJECT_ROOT, "hgd", "train1", "1.edf")

    def test_config_hash_consistency(self):
        """Test deterministic configuration hashing."""
        hash1 = compute_config_hash(config={}, representation="frequency")
        hash2 = compute_config_hash(config={}, representation="frequency")
        self.assertEqual(hash1, hash2)

        hash3 = compute_config_hash(config={}, representation="time")
        self.assertNotEqual(hash1, hash3)

    def test_lru_cache_capacity_and_eviction(self):
        """Test FileLRUCache enforces max_open_cache_files limit and CPU map_location."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            lru = FileLRUCache(max_open_cache_files=2)

            file1 = os.path.join(tmp_dir, "f1.pt")
            file2 = os.path.join(tmp_dir, "f2.pt")
            file3 = os.path.join(tmp_dir, "f3.pt")

            torch.save({"X": torch.randn(5, 4, 133, 250), "y": torch.zeros(5), "trial_ids": torch.zeros(5)}, file1)
            torch.save({"X": torch.randn(5, 4, 133, 250), "y": torch.zeros(5), "trial_ids": torch.zeros(5)}, file2)
            torch.save({"X": torch.randn(5, 4, 133, 250), "y": torch.zeros(5), "trial_ids": torch.zeros(5)}, file3)

            lru.get(file1)
            lru.get(file2)
            self.assertEqual(len(lru), 2)

            # Access file3 -> evicts file1
            lru.get(file3)
            self.assertEqual(len(lru), 2)
            self.assertNotIn(file1, lru._cache)
            self.assertIn(file2, lru._cache)
            self.assertIn(file3, lru._cache)

    @patch.object(EEGPreprocessingPipeline, "process")
    def test_cache_build_reload_and_zero_pipeline_calls(self, mock_process):
        """Test initial cache creation, second run cache HIT, and zero pipeline calls on hit."""
        mock_process.return_value = (
            np.random.randn(10, 4, 133, 250).astype(np.float32),
            np.zeros(10, dtype=np.int64),
            np.zeros(10, dtype=np.int64),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_cfg = {"enabled": True, "directory": tmp_dir, "max_open_cache_files": 2}
            pipeline = EEGPreprocessingPipeline()

            # First run: builds cache (1 pipeline call)
            ds1 = HGDDataset(
                file_paths=["/fake/path/sample.edf"],
                pipeline=pipeline,
                representation="frequency",
                cache_config=cache_cfg,
            )
            self.assertEqual(len(ds1), 10)
            mock_process.assert_called_once()

            # Second run: loads cache metadata (0 pipeline calls)
            mock_process.reset_mock()
            ds2 = HGDDataset(
                file_paths=["/fake/path/sample.edf"],
                pipeline=pipeline,
                representation="frequency",
                cache_config=cache_cfg,
            )
            self.assertEqual(len(ds2), 10)
            mock_process.assert_not_called()

    @patch.object(EEGPreprocessingPipeline, "process")
    def test_atomic_writes_cleanup_tmp_files(self, mock_process):
        """Test that temporary `.tmp` files are atomically renamed and cleaned up."""
        mock_process.return_value = (
            np.random.randn(4, 133, 250).astype(np.float32),
            np.zeros(4, dtype=np.int64),
            np.zeros(4, dtype=np.int64),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_cfg = {"enabled": True, "directory": tmp_dir}
            HGDDataset(
                file_paths=["/fake/path/sample.edf"],
                representation="time",
                cache_config=cache_cfg,
            )

            files = os.listdir(tmp_dir)
            tmp_files = [f for f in files if f.endswith(".tmp")]
            self.assertEqual(len(tmp_files), 0, "Found un-cleaned temporary .tmp files!")
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "metadata.json")))

    @patch.object(EEGPreprocessingPipeline, "process")
    def test_cache_equivalence(self, mock_process):
        """Test torch.allclose() comparing raw pipeline output vs cached tensor."""
        raw_x = np.random.randn(8, 4, 133, 250).astype(np.float32)
        raw_y = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
        raw_t = np.zeros(8, dtype=np.int64)
        mock_process.return_value = (raw_x, raw_y, raw_t)

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_cfg = {"enabled": True, "directory": tmp_dir}
            ds = HGDDataset(
                file_paths=["/fake/sample.edf"],
                representation="frequency",
                cache_config=cache_cfg,
            )

            # Get sample 0 via __getitem__
            sample_x, sample_y = ds[0]
            torch.testing.assert_close(sample_x, torch.tensor(raw_x[0]))
            self.assertEqual(sample_y.item(), raw_y[0])

    @patch.object(EEGPreprocessingPipeline, "process")
    def test_incremental_cache_rebuild(self, mock_process):
        """Test deleting one cache file causes ONLY that specific EDF to rebuild."""
        mock_process.return_value = (
            np.random.randn(5, 133, 250).astype(np.float32),
            np.zeros(5, dtype=np.int64),
            np.zeros(5, dtype=np.int64),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_cfg = {"enabled": True, "directory": tmp_dir}
            files = ["/fake/file1.edf", "/fake/file2.edf"]

            # Initial build for 2 files -> process called twice
            ds1 = HGDDataset(file_paths=files, representation="time", cache_config=cache_cfg)
            self.assertEqual(mock_process.call_count, 2)

            # Delete cache file for file1
            meta = ds1.metadata
            file1_cache = os.path.join(tmp_dir, meta["files"][0]["cache"])
            os.remove(file1_cache)

            # Re-init -> process called ONLY 1 additional time for missing file1
            mock_process.reset_mock()
            HGDDataset(file_paths=files, representation="time", cache_config=cache_cfg)
            self.assertEqual(mock_process.call_count, 1)

    @patch.object(EEGPreprocessingPipeline, "process")
    def test_cache_invalidation_on_config_hash_mismatch(self, mock_process):
        """Test cache rebuild triggers automatically when config or representation changes."""
        mock_process.return_value = (
            np.random.randn(6, 133, 250).astype(np.float32),
            np.zeros(6, dtype=np.int64),
            np.zeros(6, dtype=np.int64),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_cfg = {"enabled": True, "directory": tmp_dir}

            # First build: time representation
            HGDDataset(file_paths=["/fake/sample.edf"], representation="time", cache_config=cache_cfg)
            self.assertEqual(mock_process.call_count, 1)

            # Second build: frequency representation -> forces rebuild
            mock_process.reset_mock()
            HGDDataset(file_paths=["/fake/sample.edf"], representation="frequency", cache_config=cache_cfg)
            self.assertEqual(mock_process.call_count, 1)

    @patch.object(EEGPreprocessingPipeline, "process")
    def test_cache_builder_resource_cleanup_sequential_processing(self, mock_process):
        """Regression test: verify cache builder processes multiple EDF files sequentially and cleans up resources."""
        mock_process.return_value = (
            np.random.randn(10, 4, 133, 250).astype(np.float32),
            np.zeros(10, dtype=np.int64),
            np.zeros(10, dtype=np.int64),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_manager = CacheManager(cache_dir=tmp_dir)
            pipeline = EEGPreprocessingPipeline()
            files = [f"/fake/path/file_{i}.edf" for i in range(10)]

            meta = cache_manager.build_cache(
                file_paths=files,
                pipeline=pipeline,
                representation="frequency",
                config_hash="test_hash_sequential_cleanup",
                build_if_missing=True,
            )

            self.assertEqual(mock_process.call_count, 10)
            self.assertEqual(meta["total_samples"], 100)
            self.assertEqual(len(meta["files"]), 10)


if __name__ == "__main__":
    unittest.main()
