"""
Modular Dataset Cache Management and Per-EDF LRU Caching System.

Provides CacheManager, FileLRUCache, configuration hashing, atomic disk I/O,
and metadata-driven index resolution for High-Gamma Dataset (HGD) EEG processing.
Phase 10 Patch v0.10.4: Production-Grade Lazy HGD Dataset Layer.
"""

import os
import json
import time
import gc
import hashlib
import logging
import tempfile
from collections import OrderedDict
from typing import List, Dict, Tuple, Optional, Any, Union

import torch
import numpy as np

logger = logging.getLogger(__name__)

CACHE_VERSION = 2


def compute_config_hash(
    config: Optional[Union[Dict[str, Any], Any]] = None,
    representation: str = "frequency",
) -> str:
    """
    Compute a deterministic SHA256 configuration hash based on preprocessing parameters,
    representation mode, sub-bands, window size, stride, normalization, and sampling rate.

    Args:
        config: Configuration dictionary or PreprocessingConfig object.
        representation: "time" or "frequency".

    Returns:
        str: 16-character SHA256 hex string.
    """
    if hasattr(config, "to_dict"):
        cfg_dict = config.to_dict()
    elif isinstance(config, dict):
        cfg_dict = config
    else:
        cfg_dict = {}

    # Extract critical pipeline fields to ensure hash accuracy
    norm_cfg = cfg_dict.get("normalization", {})
    win_cfg = cfg_dict.get("windowing", {})
    filt_cfg = cfg_dict.get("filtering", {})
    freq_cfg = cfg_dict.get("frequency", {})

    key_dict = {
        "cache_version": CACHE_VERSION,
        "representation": representation,
        "sampling_rate": cfg_dict.get("sampling_rate", 250.0),
        "target_channels": cfg_dict.get("target_channels", None),
        "filtering": {
            "l_freq": filt_cfg.get("l_freq", 4.0),
            "h_freq": filt_cfg.get("h_freq", 125.0),
            "notch_freqs": filt_cfg.get("notch_freqs", [50.0]),
        },
        "normalization": {
            "method": norm_cfg.get("method", "zscore"),
            "per_channel": norm_cfg.get("per_channel", True),
        },
        "windowing": {
            "window_size_samples": win_cfg.get("window_size_samples", 250),
            "stride_samples": win_cfg.get("stride_samples", 50),
            "tmin": win_cfg.get("tmin", 0.0),
            "tmax": win_cfg.get("tmax", 4.0),
        },
        "frequency": {
            "enabled": freq_cfg.get("enabled", True),
            "bands": freq_cfg.get("bands", None),
        },
    }

    serialized = json.dumps(key_dict, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


class FileLRUCache:
    """
    Least Recently Used (LRU) cache for pre-processed per-EDF trial `.pt` PyTorch tensor files.
    Ensures that only a maximum of `max_open_cache_files` trial tensors reside in RAM simultaneously.
    Tensors are loaded explicitly onto CPU via `map_location='cpu'`.
    """

    def __init__(self, max_open_cache_files: int = 2):
        """
        Initialize FileLRUCache.

        Args:
            max_open_cache_files (int): Maximum number of EDF `.pt` files kept in memory.
        """
        self.max_open_cache_files = max(1, max_open_cache_files)
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()

    def get(self, cache_file_path: str) -> Dict[str, Any]:
        """
        Retrieve trial dictionary from cache, loading from disk on cache miss.

        Args:
            cache_file_path (str): Absolute path to `.pt` cache file.

        Returns:
            Dict[str, Any]: Dictionary containing "trials", "labels", "trial_ids".
        """
        if cache_file_path in self._cache:
            self._cache.move_to_end(cache_file_path)
            return self._cache[cache_file_path]

        if not os.path.exists(cache_file_path):
            raise FileNotFoundError(f"Cached tensor file missing at: {cache_file_path}")

        # Load tensor on CPU strictly
        data = torch.load(cache_file_path, map_location="cpu", weights_only=False)

        # Integrity verification
        if data.get("cache_version") != CACHE_VERSION:
            raise ValueError(f"Cache file {cache_file_path} is incompatible version {data.get('cache_version')}")

        # Evict oldest entry if capacity reached
        if len(self._cache) >= self.max_open_cache_files:
            evicted_path, _ = self._cache.popitem(last=False)
            logger.debug(f"Evicted cache file from LRU memory: {os.path.basename(evicted_path)}")

        self._cache[cache_file_path] = data
        return data

    def clear(self) -> None:
        """Clear all loaded tensors from LRU memory."""
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


class CacheManager:
    """
    Manages trial-level cache building, validation, incremental rebuilding, atomic I/O,
    and metadata index management for HGD EEG pre-processed files.
    """

    def __init__(self, cache_dir: str = "outputs/cache"):
        self.cache_dir = cache_dir
        self.metadata_path = os.path.join(self.cache_dir, "metadata.json")

    @staticmethod
    def _safe_replace(tmp_path: str, target_path: str, retries: int = 5, delay: float = 0.1) -> None:
        """Helper to replace files atomically with retry logic for Windows file locking."""
        for attempt in range(retries):
            try:
                os.replace(tmp_path, target_path)
                return
            except PermissionError as e:
                if attempt == retries - 1:
                    raise e
                time.sleep(delay)

    @staticmethod
    def atomic_torch_save(obj: Any, target_path: str) -> None:
        """Save a PyTorch object atomically using a temporary file and rename."""
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        tmp_path = target_path + ".tmp"
        with open(tmp_path, "wb") as f:
            torch.save(obj, f)
        CacheManager._safe_replace(tmp_path, target_path)

    @staticmethod
    def atomic_json_save(obj: Dict[str, Any], target_path: str) -> None:
        """Save a dictionary as JSON atomically using a temporary file and rename."""
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        tmp_path = target_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
        CacheManager._safe_replace(tmp_path, target_path)

    def load_metadata(self) -> Optional[Dict[str, Any]]:
        """Load metadata.json if it exists and is valid JSON."""
        if not os.path.exists(self.metadata_path):
            return None
        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.warning(f"Failed to read metadata.json from {self.metadata_path}: {e}")
            return None

    def purge_legacy_cache(self) -> None:
        """Phase 0 Migration: Remove legacy CACHE_VERSION 1 files to prevent stale artifacts."""
        if os.path.exists(self.cache_dir):
            logger.info("Phase 0 Migration: Purging legacy window-cache files from cache directory...")
            for fname in os.listdir(self.cache_dir):
                fpath = os.path.join(self.cache_dir, fname)
                try:
                    if os.path.isfile(fpath):
                        os.remove(fpath)
                except Exception as e:
                    logger.warning(f"Could not remove stale file {fpath}: {e}")

    def validate_cache(
        self,
        file_paths: List[str],
        representation: str,
        config_hash: str,
    ) -> Tuple[bool, List[str]]:
        """
        Validate whether trial cache exists and matches configuration, version, and source mtimes.

        Args:
            file_paths: List of EDF file paths expected in dataset.
            representation: "time" or "frequency".
            config_hash: Current computed configuration SHA256 hash.

        Returns:
            Tuple[bool, List[str]]:
                - valid (bool): True if ALL files have valid, matching up-to-date cache.
                - invalid_files (List[str]): List of EDF file paths needing build/rebuild.
        """
        meta = self.load_metadata()
        if meta is None:
            logger.info("Cache Status : MISS (metadata.json missing)")
            return False, list(file_paths)

        if meta.get("cache_version") != CACHE_VERSION:
            logger.info(
                f"Cache version mismatch. Expected version={CACHE_VERSION}, found version={meta.get('cache_version')}."
            )
            self.purge_legacy_cache()
            return False, list(file_paths)

        if meta.get("config_hash") != config_hash:
            logger.info(
                f"Cache configuration hash mismatch.\nExpected: {config_hash}\nFound: {meta.get('config_hash')}\nRebuilding..."
            )
            return False, list(file_paths)

        if meta.get("representation") != representation:
            logger.info(
                f"Cache representation mismatch. Expected '{representation}', found '{meta.get('representation')}'."
            )
            return False, list(file_paths)

        files_meta = {item["edf_path"]: item for item in meta.get("files", [])}
        invalid_files = []

        for f_path in file_paths:
            abs_path = os.path.abspath(f_path)
            item = files_meta.get(abs_path)
            if not item:
                invalid_files.append(f_path)
                continue

            cache_file = os.path.join(self.cache_dir, item["cache"])
            if not os.path.exists(cache_file):
                invalid_files.append(f_path)
                continue

            # Check EDF modification time
            if os.path.exists(abs_path):
                current_mtime = int(os.path.getmtime(abs_path))
                if item.get("edf_mtime") != current_mtime:
                    logger.info(f"Source EDF file modified: {f_path}. Tagged for incremental rebuild.")
                    invalid_files.append(f_path)

        all_valid = len(invalid_files) == 0
        if all_valid:
            logger.info("Cache Status : HIT")
        else:
            logger.info(f"Cache Status : MISS ({len(invalid_files)} files require build/rebuild)")

        return all_valid, invalid_files

    def build_cache(
        self,
        file_paths: List[str],
        pipeline: Any,
        representation: str,
        config_hash: str,
        build_if_missing: bool = True,
        progress_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Build or incrementally update per-EDF trial cache `.pt` files and `metadata.json`.

        Args:
            file_paths: List of EDF file paths to include in cache.
            pipeline: EEGPreprocessingPipeline instance for trial processing.
            representation: "time" or "frequency".
            config_hash: Current SHA256 configuration hash string.
            build_if_missing: If False and cache missing, raises RuntimeError.
            progress_callback: Optional callback for progress monitoring.

        Returns:
            Dict[str, Any]: Updated metadata dictionary.
        """
        valid, invalid_files = self.validate_cache(file_paths, representation, config_hash)

        if valid and os.path.exists(self.metadata_path):
            return self.load_metadata()

        if not build_if_missing and len(invalid_files) > 0:
            raise RuntimeError(
                f"Dataset cache missing or invalid for {len(invalid_files)} files, "
                f"and build_if_missing=False.\nPlease run: python scripts/precompute_dataset.py"
            )

        os.makedirs(self.cache_dir, exist_ok=True)

        # Existing metadata files mapping
        existing_meta = self.load_metadata() or {}
        existing_files_meta = {
            item["edf_path"]: item for item in existing_meta.get("files", [])
        }

        # Extract representative window size and stride
        win_size = 250
        stride = 50
        if hasattr(pipeline, "config") and pipeline.config is not None:
            cfg = pipeline.config
            if hasattr(cfg, "window_size"):
                win_size = int(getattr(cfg, "window_size", 250))
            elif hasattr(cfg, "windowing") and hasattr(cfg.windowing, "window_size_samples"):
                win_size = int(getattr(cfg.windowing, "window_size_samples", 250))

            if hasattr(cfg, "window_stride"):
                stride = int(getattr(cfg, "window_stride", 50))
            elif hasattr(cfg, "windowing") and hasattr(cfg.windowing, "stride_samples"):
                stride = int(getattr(cfg.windowing, "stride_samples", 50))

        # Process each EDF requiring build
        num_invalid = len(invalid_files)
        for idx, f_path in enumerate(invalid_files):
            abs_path = os.path.abspath(f_path)
            logger.info(f"[{idx+1}/{num_invalid}] Preprocessing EDF trials for cache: {f_path}")

            if hasattr(pipeline, "process_trials"):
                X_trials, y_trials, t_ids = pipeline.process_trials(f_path, representation=representation)
            else:
                X_trials, y_trials, t_ids = pipeline.process(f_path, representation=representation)

            X_tensor = torch.tensor(X_trials, dtype=torch.float32)
            y_tensor = torch.tensor(y_trials, dtype=torch.long)
            t_tensor = torch.tensor(t_ids, dtype=torch.long)

            mtime = int(os.path.getmtime(abs_path)) if os.path.exists(abs_path) else 0

            # Derive unique cache filename
            split_dir = os.path.basename(os.path.dirname(f_path))
            base_name = os.path.splitext(os.path.basename(f_path))[0]
            dir_hash = hashlib.md5(os.path.dirname(abs_path).encode("utf-8")).hexdigest()[:6]
            cache_filename = f"{split_dir}_{base_name}_{representation}_{dir_hash}_trials.pt"
            cache_filepath = os.path.join(self.cache_dir, cache_filename)

            # Save per-trial data dict
            data_dict = {
                "trials": X_tensor,
                "labels": y_tensor,
                "trial_ids": t_tensor,
                "edf_path": abs_path,
                "edf_mtime": mtime,
                "config_hash": config_hash,
                "cache_version": CACHE_VERSION,
                "representation": representation,
                "dtype": str(X_tensor.dtype),
            }
            self.atomic_torch_save(data_dict, cache_filepath)

            # Precalculate cumulative trial start window indices for robust O(log N) binary search
            n_trials = len(y_trials)
            trial_length_samples = X_trials.shape[-1]
            trial_start_indices = [0]
            cumulative_windows = 0

            for trial_idx in range(n_trials):
                num_windows_in_trial = len(range(0, trial_length_samples - win_size + 1, stride))
                cumulative_windows += num_windows_in_trial
                trial_start_indices.append(cumulative_windows)

            existing_files_meta[abs_path] = {
                "edf_path": abs_path,
                "cache": cache_filename,
                "num_trials": n_trials,
                "trial_length_samples": trial_length_samples,
                "trial_start_indices": trial_start_indices,
                "total_windows": cumulative_windows,
                "edf_mtime": mtime,
                "config_hash": config_hash,
                "tensor_shape": list(X_tensor.shape),
            }

            # Save metadata index incrementally so partial builds are safely saved
            self._save_metadata_index(file_paths, existing_files_meta, pipeline, representation, config_hash)

            # Explicitly release large arrays and intermediate tensors after each EDF
            del X_trials, y_trials, t_ids, X_tensor, y_tensor, t_tensor, data_dict
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if progress_callback:
                progress_callback(idx + 1, num_invalid)

        return self._save_metadata_index(file_paths, existing_files_meta, pipeline, representation, config_hash)

    def _save_metadata_index(
        self,
        file_paths: List[str],
        existing_files_meta: Dict[str, Any],
        pipeline: Any,
        representation: str,
        config_hash: str,
    ) -> Dict[str, Any]:
        """Helper to build and save metadata.json index atomically."""
        ordered_files_meta = []
        global_offset = 0

        for f_path in file_paths:
            abs_path = os.path.abspath(f_path)
            if abs_path in existing_files_meta:
                meta_entry = dict(existing_files_meta[abs_path])
                n_windows = meta_entry["total_windows"]
                meta_entry["start_index"] = global_offset
                meta_entry["end_index"] = global_offset + n_windows - 1 if n_windows > 0 else global_offset
                global_offset += n_windows
                ordered_files_meta.append(meta_entry)

        win_size = 250
        stride = 50
        if hasattr(pipeline, "config") and pipeline.config is not None:
            cfg = pipeline.config
            if hasattr(cfg, "window_size"):
                win_size = int(getattr(cfg, "window_size", 250))
            elif hasattr(cfg, "windowing") and hasattr(cfg.windowing, "window_size_samples"):
                win_size = int(getattr(cfg.windowing, "window_size_samples", 250))

            if hasattr(cfg, "window_stride"):
                stride = int(getattr(cfg, "window_stride", 50))
            elif hasattr(cfg, "windowing") and hasattr(cfg.windowing, "stride_samples"):
                stride = int(getattr(cfg.windowing, "stride_samples", 50))

        master_metadata = {
            "cache_version": CACHE_VERSION,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_root": os.path.commonpath([os.path.abspath(p) for p in file_paths]) if file_paths else "",
            "representation": representation,
            "config_hash": config_hash,
            "window_size": win_size,
            "stride": stride,
            "total_samples": global_offset,
            "files": ordered_files_meta,
        }

        self.atomic_json_save(master_metadata, self.metadata_path)
        logger.info(f"Updated metadata index saved to: {self.metadata_path} (Total samples: {global_offset})")
        return master_metadata
