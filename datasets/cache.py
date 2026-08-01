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
    Ensures that trial tensors reside in RAM up to a configurable RAM budget or file count limit.
    Tensors are loaded explicitly onto CPU via `map_location='cpu'`.
    Includes comprehensive instrumentation tracking (hits, misses, load count, evictions, RAM usage).
    """

    def __init__(
        self,
        max_open_cache_files: int = 14,
        max_ram_gb: Union[float, str] = "auto",
        memory_budget_fraction: float = 0.25,
    ):
        """
        Initialize FileLRUCache.

        Args:
            max_open_cache_files (int): Maximum number of EDF `.pt` files kept in memory.
            max_ram_gb (Union[float, str]): RAM limit in GB or "auto" to derive from system RAM.
            memory_budget_fraction (float): Fraction of available system RAM to allocate if max_ram_gb="auto".
        """
        self.max_open_cache_files = max(1, max_open_cache_files) if max_open_cache_files else 128
        self.max_ram_gb = max_ram_gb
        self.memory_budget_fraction = memory_budget_fraction

        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._entry_bytes: Dict[str, int] = {}
        self._current_bytes: int = 0

        # Instrumentation Metrics
        self.hits: int = 0
        self.misses: int = 0
        self.load_count: int = 0
        self.evictions: int = 0
        self.peak_open_files: int = 0
        self.peak_bytes: int = 0

        self.ram_budget_bytes = self._resolve_ram_budget()

    def _resolve_ram_budget(self) -> int:
        if isinstance(self.max_ram_gb, (int, float)) and self.max_ram_gb > 0:
            return int(self.max_ram_gb * (1024 ** 3))
        try:
            import psutil
            total_ram = psutil.virtual_memory().total
        except Exception:
            total_ram = 8 * (1024 ** 3)
        return int(total_ram * self.memory_budget_fraction)

    @staticmethod
    def _estimate_dict_bytes(data: Dict[str, Any]) -> int:
        total_bytes = 0
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, torch.Tensor):
                    total_bytes += v.element_size() * v.nelement()
                elif isinstance(v, np.ndarray):
                    total_bytes += v.nbytes
        return total_bytes

    def get(self, cache_file_path: str) -> Dict[str, Any]:
        """
        Retrieve trial dictionary from cache, loading from disk on cache miss.

        Args:
            cache_file_path (str): Path to `.pt` cache file.

        Returns:
            Dict[str, Any]: Dictionary containing "trials", "labels", "trial_ids".
        """
        abs_key = os.path.abspath(cache_file_path)

        if abs_key in self._cache:
            self.hits += 1
            self._cache.move_to_end(abs_key)
            return self._cache[abs_key]

        self.misses += 1
        if not os.path.exists(abs_key):
            raise FileNotFoundError(f"Cached tensor file missing at: {abs_key}")

        # Load tensor on CPU strictly
        data = torch.load(abs_key, map_location="cpu", weights_only=False)
        self.load_count += 1

        # Integrity verification
        if data.get("cache_version") != CACHE_VERSION:
            raise ValueError(f"Cache file {abs_key} is incompatible version {data.get('cache_version')}")

        entry_size = self._estimate_dict_bytes(data)

        # Evict oldest entries if capacity or RAM budget reached
        while len(self._cache) > 0 and (
            len(self._cache) >= self.max_open_cache_files
            or (self._current_bytes + entry_size > self.ram_budget_bytes)
        ):
            evicted_path, evicted_data = self._cache.popitem(last=False)
            evicted_bytes = self._entry_bytes.pop(evicted_path, 0)
            self._current_bytes = max(0, self._current_bytes - evicted_bytes)
            self.evictions += 1
            logger.debug(f"Evicted cache file from LRU memory: {os.path.basename(evicted_path)}")

        self._cache[abs_key] = data
        self._entry_bytes[abs_key] = entry_size
        self._current_bytes += entry_size

        self.peak_open_files = max(self.peak_open_files, len(self._cache))
        self.peak_bytes = max(self.peak_bytes, self._current_bytes)

        return data

    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def get_stats(self) -> Dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio": self.hit_ratio,
            "hit_ratio_pct": self.hit_ratio * 100.0,
            "load_count": self.load_count,
            "evictions": self.evictions,
            "current_open_files": len(self._cache),
            "peak_open_files": self.peak_open_files,
            "max_open_cache_files": self.max_open_cache_files,
            "current_ram_mb": self._current_bytes / (1024 ** 2),
            "peak_ram_mb": self.peak_bytes / (1024 ** 2),
            "ram_budget_mb": self.ram_budget_bytes / (1024 ** 2),
        }

    def print_stats(self) -> None:
        stats = self.get_stats()
        msg = (
            f"========================\n"
            f"Cache Statistics\n"
            f"------------------------\n"
            f"Hits        : {stats['hits']}\n"
            f"Misses      : {stats['misses']}\n"
            f"Hit Ratio   : {stats['hit_ratio_pct']:.2f}%\n"
            f"Files Loaded: {stats['load_count']}\n"
            f"Evictions   : {stats['evictions']}\n"
            f"Open Files  : {stats['current_open_files']} (Peak: {stats['peak_open_files']})\n"
            f"RAM Memory  : {stats['current_ram_mb']:.1f} MB (Peak: {stats['peak_ram_mb']:.1f} MB / Budget: {stats['ram_budget_mb']:.1f} MB)\n"
            f"========================"
        )
        logger.info("\n" + msg)
        print("\n" + msg)

    def reset_stats(self) -> None:
        self.hits = 0
        self.misses = 0
        self.load_count = 0
        self.evictions = 0
        self.peak_open_files = len(self._cache)
        self.peak_bytes = self._current_bytes

    def clear(self) -> None:
        """Clear all loaded tensors from LRU memory."""
        self._cache.clear()
        self._entry_bytes.clear()
        self._current_bytes = 0
        self.reset_stats()

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
            except (PermissionError, FileNotFoundError) as e:
                if os.path.exists(target_path) and not os.path.exists(tmp_path):
                    return
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

    def purge_stale_duplicates(self, active_cache_filenames: Optional[List[str]] = None) -> None:
        """Scan cache directory and delete stale/duplicate, tmp, or orphan .pt files."""
        if not os.path.exists(self.cache_dir):
            return
        active_set = set(active_cache_filenames) if active_cache_filenames else set()
        for fname in os.listdir(self.cache_dir):
            fpath = os.path.join(self.cache_dir, fname)
            if fname.endswith(".pt.tmp"):
                try:
                    os.remove(fpath)
                except Exception:
                    pass
            elif fname.endswith(".pt"):
                if active_cache_filenames is not None and fname not in active_set:
                    try:
                        os.remove(fpath)
                        logger.info(f"Purged stale/duplicate cache artifact: {fname}")
                    except Exception as e:
                        logger.warning(f"Failed to delete stale cache file {fpath}: {e}")

    def validate_cache(
        self,
        file_paths: List[str],
        representation: str,
        config_hash: str,
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Validate whether trial cache exists and matches all 8 integrity checks.

        Args:
            file_paths: List of EDF file paths expected in dataset.
            representation: "time" or "frequency".
            config_hash: Current computed configuration SHA256 hash.

        Returns:
            Tuple[bool, List[str], List[str]]:
                - valid (bool): True if ALL files pass 8-point validation.
                - invalid_files (List[str]): List of EDF file paths needing build/rebuild.
                - invalid_reasons (List[str]): Detailed diagnostic reasons for each miss.
        """
        meta = self.load_metadata()
        if meta is None:
            logger.info("Cache Status : MISS (metadata.json missing)")
            return False, list(file_paths), ["metadata.json missing"]

        # Check 4 & 5: Global metadata version, hash, and representation
        if meta.get("cache_version") != CACHE_VERSION:
            reason = f"Cache version mismatch. Expected version={CACHE_VERSION}, found={meta.get('cache_version')}"
            logger.info(f"Cache Status : MISS\nReason       : {reason}")
            self.purge_legacy_cache()
            return False, list(file_paths), [reason] * len(file_paths)

        if meta.get("config_hash") != config_hash:
            reason = f"Config hash mismatch. Expected: {config_hash}, Found: {meta.get('config_hash')}"
            logger.info(f"Cache Status : MISS\nReason       : {reason}")
            return False, list(file_paths), [reason] * len(file_paths)

        if meta.get("representation") != representation:
            reason = f"Representation mismatch. Expected: '{representation}', Found: '{meta.get('representation')}'"
            logger.info(f"Cache Status : MISS\nReason       : {reason}")
            return False, list(file_paths), [reason] * len(file_paths)

        files_meta = {item["edf_path"]: item for item in meta.get("files", [])}
        invalid_files = []
        invalid_reasons = []

        for f_path in file_paths:
            abs_path = os.path.abspath(f_path)
            item = files_meta.get(abs_path)

            # Check 3: Metadata entry existence & identity
            if not item:
                invalid_files.append(f_path)
                invalid_reasons.append(f"Missing metadata entry for EDF: {os.path.basename(f_path)}")
                continue

            cache_file = os.path.join(self.cache_dir, item["cache"])

            # Check 8: Actual .pt file existence
            if not os.path.exists(cache_file):
                invalid_files.append(f_path)
                invalid_reasons.append(f"Missing cache tensor file: {item['cache']}")
                continue

            # Check 1 & 2: EDF file existence & modification time check (if source file exists)
            if os.path.exists(abs_path):
                current_mtime = int(os.path.getmtime(abs_path))
                if item.get("edf_mtime") != current_mtime:
                    invalid_files.append(f_path)
                    invalid_reasons.append(f"Source EDF file modified (mtime mismatch for {os.path.basename(f_path)})")
                    continue

            # Check 6 & 7: Tensor integrity check
            try:
                data = torch.load(cache_file, map_location="cpu", weights_only=False)
                if not isinstance(data, dict):
                    invalid_files.append(f_path)
                    invalid_reasons.append(f"Corrupted cache file structure: {item['cache']}")
                    continue

                missing_keys = [k for k in ("trials", "labels", "trial_ids") if k not in data]
                if missing_keys:
                    invalid_files.append(f_path)
                    invalid_reasons.append(f"Missing required tensor keys {missing_keys} in {item['cache']}")
                    continue

                if data.get("config_hash") != config_hash:
                    invalid_files.append(f_path)
                    invalid_reasons.append(f"Internal config hash mismatch in {item['cache']}")
                    continue

            except Exception as e:
                invalid_files.append(f_path)
                invalid_reasons.append(f"Failed to load cache file {item['cache']}: {e}")
                continue

        all_valid = len(invalid_files) == 0
        if all_valid:
            logger.info(f"Cache Status : HIT (Reusing {len(file_paths)} valid cached trial files)")
        else:
            first_reason = invalid_reasons[0] if invalid_reasons else "Unknown validation failure"
            logger.info(
                f"Cache Status : MISS ({len(invalid_files)} files require build/rebuild)\n"
                f"Reason       : {first_reason}"
            )

        return all_valid, invalid_files, invalid_reasons

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
        Applies Smart Cache Reuse, 8-point validation, and duplicate cleanup.

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
        valid, invalid_files, invalid_reasons = self.validate_cache(file_paths, representation, config_hash)

        existing_meta = self.load_metadata() or {}
        existing_files_meta = {
            item["edf_path"]: item for item in existing_meta.get("files", [])
        }

        if valid and os.path.exists(self.metadata_path):
            active_cache_files = [item["cache"] for item in existing_meta.get("files", [])]
            self.purge_stale_duplicates(active_cache_files)
            return existing_meta

        if not build_if_missing and len(invalid_files) > 0:
            raise RuntimeError(
                f"Dataset cache missing or invalid for {len(invalid_files)} files, "
                f"and build_if_missing=False.\nPlease run: python scripts/precompute_dataset.py"
            )

        os.makedirs(self.cache_dir, exist_ok=True)
        active_valid_files = [item["cache"] for item in existing_files_meta.values() if os.path.exists(os.path.join(self.cache_dir, item.get("cache", "")))]
        self.purge_stale_duplicates(active_valid_files)

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
            reason = invalid_reasons[idx] if idx < len(invalid_reasons) else "Missing or outdated cache"
            logger.info(f"[{idx+1}/{num_invalid}] Rebuilding cache for {f_path} (Reason: {reason})")

            if hasattr(pipeline, "process_trials"):
                X_trials, y_trials, t_ids = pipeline.process_trials(f_path, representation=representation)
            else:
                X_trials, y_trials, t_ids = pipeline.process(f_path, representation=representation)

            X_tensor = torch.tensor(X_trials, dtype=torch.float16)
            y_tensor = torch.tensor(y_trials, dtype=torch.long)
            t_tensor = torch.tensor(t_ids, dtype=torch.long)

            mtime = int(os.path.getmtime(abs_path)) if os.path.exists(abs_path) else 0

            # Derive clean cache filename: train1_13_frequency_trials.pt
            split_dir = os.path.basename(os.path.dirname(f_path))
            base_name = os.path.splitext(os.path.basename(f_path))[0]
            cache_filename = f"{split_dir}_{base_name}_{representation}_trials.pt"
            cache_filepath = os.path.join(self.cache_dir, cache_filename)

            # Save per-trial data dict with explicit metadata header
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

        updated_metadata = self._save_metadata_index(file_paths, existing_files_meta, pipeline, representation, config_hash)

        # Cleanup stale duplicate .pt files
        active_cache_files = [item["cache"] for item in updated_metadata.get("files", [])]
        self.purge_stale_duplicates(active_cache_files)

        return updated_metadata

    def _save_metadata_index(
        self,
        file_paths: List[str],
        existing_files_meta: Dict[str, Any],
        pipeline: Any,
        representation: str,
        config_hash: str,
    ) -> Dict[str, Any]:
        ordered_files_meta = []
        global_offset = 0

        all_paths = list(file_paths)
        for p in existing_files_meta.keys():
            if p not in all_paths:
                all_paths.append(p)

        for abs_path in all_paths:
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
