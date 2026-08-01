"""
PyTorch HGDDataset Implementation.

Provides HGDDataset wrapping windowed motor imagery EEG signals for PyTorch
DataLoader integration. Supports both time-domain and frequency-aware multi-band representations.
Phase 10 Patch v0.10.4: Production-Grade Lazy HGD Dataset Layer.
"""

import os
import logging
from bisect import bisect_right
from typing import List, Union, Optional, Tuple, Dict, Any

import torch
from torch.utils.data import Dataset

from datasets.pipeline import EEGPreprocessingPipeline, PreprocessingConfig
from datasets.windowing import extract_single_window_from_trial
from datasets.cache import (
    CacheManager,
    FileLRUCache,
    compute_config_hash,
    CACHE_VERSION,
)

logger = logging.getLogger(__name__)


class HGDDataset(Dataset):
    """
    PyTorch Dataset wrapper for High Gamma Dataset (HGD) Motor Imagery EEG samples.

    Uses metadata-driven index mapping (`metadata.json`) and lazy trial-level caching
    via FileLRUCache with on-the-fly window extraction in `__getitem__()` to keep
    RAM footprint low (~50-150 MB RAM) and cache storage footprint minimal (~2-4 GB).
    """

    def __init__(
        self,
        file_paths: Union[str, List[str]],
        pipeline: Optional[EEGPreprocessingPipeline] = None,
        config: Optional[Union[PreprocessingConfig, str, Dict[str, Any]]] = None,
        representation: str = "time",
        cache_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize HGDDataset.

        Args:
            file_paths: Path to a single EDF file or list of EDF file paths.
            pipeline: Pre-configured EEGPreprocessingPipeline instance (optional).
            config: Configuration for pipeline if pipeline is not provided (optional).
            representation: "time" for (N, Channels, Samples) or "frequency" for (N, Bands, Channels, Samples).
            cache_config: Dictionary with cache settings:
                - enabled (bool): Default True.
                - directory (str): Path to cache directory (default "outputs/cache").
                - max_open_cache_files (int): Maximum open `.pt` files in RAM (default 2).
                - build_if_missing (bool): Whether to build cache on the fly (default True).
        """
        if representation not in ("time", "frequency"):
            raise ValueError(f"Unknown representation mode '{representation}'. Supported modes: 'time', 'frequency'")

        self.representation = representation
        self.cache_config = cache_config or {}

        if isinstance(file_paths, str):
            self.file_paths = [file_paths]
        else:
            self.file_paths = list(file_paths)

        if pipeline is not None:
            self.pipeline = pipeline
        else:
            self.pipeline = EEGPreprocessingPipeline(config=config)

        self.cache_enabled = self.cache_config.get("enabled", True)
        self.cache_dir = self.cache_config.get("directory", "outputs/cache")
        self.max_open_cache_files = self.cache_config.get("max_open_cache_files", 14)
        self.max_ram_gb = self.cache_config.get("max_ram_gb", "auto")
        self.memory_budget_fraction = self.cache_config.get("memory_budget_fraction", 0.25)
        self.build_if_missing = self.cache_config.get("build_if_missing", True)

        self.config_hash = compute_config_hash(self.pipeline.config, self.representation)
        self.cache_manager = CacheManager(cache_dir=self.cache_dir)

        # Build/validate cache and load metadata index ONLY (reads ZERO .pt files into RAM)
        self.metadata = self.cache_manager.build_cache(
            file_paths=self.file_paths,
            pipeline=self.pipeline,
            representation=self.representation,
            config_hash=self.config_hash,
            build_if_missing=self.build_if_missing,
        )

        all_entries = self.metadata.get("files", [])
        requested_abs_paths = set(os.path.abspath(p) for p in self.file_paths)
        self._file_entries = [e for e in all_entries if os.path.abspath(e["edf_path"]) in requested_abs_paths]

        global_offset = 0
        self._start_indices = []
        self._end_indices = []
        for e in self._file_entries:
            n_win = e["total_windows"]
            self._start_indices.append(global_offset)
            self._end_indices.append(global_offset + n_win - 1 if n_win > 0 else global_offset)
            global_offset += n_win
        self._total_samples = global_offset

        # Initialize LRU Cache for per-EDF sample retrieval
        self._lru_cache = FileLRUCache(
            max_open_cache_files=self.max_open_cache_files,
            max_ram_gb=self.max_ram_gb,
            memory_budget_fraction=self.memory_budget_fraction,
        )

        # Lazy concatenated tensors cache for legacy compatibility properties (.X, .y)
        self._cached_concat_X: Optional[torch.Tensor] = None
        self._cached_concat_y: Optional[torch.Tensor] = None
        self._cached_concat_trial_ids: Optional[torch.Tensor] = None

        logger.info(
            f"HGDDataset initialized. Total samples: {self._total_samples}, "
            f"Cached files: {len(self._file_entries)}, Config hash: {self.config_hash}"
        )

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get underlying LRU cache instrumentation statistics."""
        return self._lru_cache.get_stats()

    def print_cache_stats(self) -> None:
        """Print underlying LRU cache instrumentation statistics."""
        self._lru_cache.print_stats()

    def reset_cache_stats(self) -> None:
        """Reset underlying LRU cache instrumentation statistics."""
        self._lru_cache.reset_stats()

    def __len__(self) -> int:
        """Return total number of cropped window samples."""
        return self._total_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get single sample-label pair using binary search trial mapping and lazy window extraction.

        Args:
            idx (int): Global sample index.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Sample tensor (shape: [Channels, window_size] or [Bands, Channels, window_size])
                - Label scalar
        """
        if idx < 0 or idx >= self._total_samples:
            raise IndexError(f"Index {idx} out of range for dataset with {self._total_samples} samples.")

        # Fast O(log N) binary search mapping global window index -> file entry
        file_idx = bisect_right(self._start_indices, idx) - 1
        file_entry = self._file_entries[file_idx]
        local_window_idx = idx - file_entry["start_index"]

        cache_path = os.path.join(self.cache_dir, file_entry["cache"])
        data = self._lru_cache.get(cache_path)

        if "trials" in data:
            trial_start_indices = file_entry["trial_start_indices"]
            trial_idx = bisect_right(trial_start_indices, local_window_idx) - 1
            local_sample_idx = local_window_idx - trial_start_indices[trial_idx]

            window_size = self.metadata.get("window_size", 250)
            stride = self.metadata.get("stride", 50)
            start_sample = local_sample_idx * stride

            trial_tensor = data["trials"][trial_idx]
            label = data["labels"][trial_idx]

            eps = getattr(self.pipeline.config, "eps", 1e-6) if hasattr(self.pipeline, "config") else 1e-6
            window_tensor = extract_single_window_from_trial(
                trial=trial_tensor,
                start_sample=start_sample,
                window_size=window_size,
                normalize=True,
                eps=eps,
            )
            return window_tensor, label
        else:
            return data["X"][local_window_idx], data["y"][local_window_idx]

    def _load_all_for_legacy_property(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Helper method to load and concatenate all tensors for legacy property access."""
        all_X, all_y, all_t = [], [], []
        window_size = self.metadata.get("window_size", 250)
        stride = self.metadata.get("stride", 50)
        eps = getattr(self.pipeline.config, "eps", 1e-6) if hasattr(self.pipeline, "config") else 1e-6

        for entry in self._file_entries:
            cache_path = os.path.join(self.cache_dir, entry["cache"])
            data = torch.load(cache_path, map_location="cpu", weights_only=False)

            if "trials" in data:
                trials = data["trials"]
                labels = data["labels"]
                t_ids = data["trial_ids"]
                trial_start_indices = entry["trial_start_indices"]

                for t_idx in range(len(labels)):
                    n_windows = trial_start_indices[t_idx + 1] - trial_start_indices[t_idx]
                    for w_sub in range(n_windows):
                        s_sample = w_sub * stride
                        w_tensor = extract_single_window_from_trial(
                            trial=trials[t_idx],
                            start_sample=s_sample,
                            window_size=window_size,
                            normalize=True,
                            eps=eps,
                        )
                        all_X.append(w_tensor)
                        all_y.append(labels[t_idx])
                        all_t.append(t_ids[t_idx])
            else:
                all_X.append(data["X"])
                all_y.append(data["y"])
                all_t.append(data["trial_ids"])

        if len(all_X) > 0:
            if isinstance(all_X[0], torch.Tensor) and all_X[0].ndim > 0:
                X_concat = torch.stack(all_X, dim=0) if all_X[0].ndim != 4 else torch.cat(all_X, dim=0)
            else:
                X_concat = torch.tensor(all_X)

            if isinstance(all_y[0], torch.Tensor):
                y_concat = torch.stack(all_y, dim=0) if all_y[0].ndim == 0 else torch.cat(all_y, dim=0)
            else:
                y_concat = torch.tensor(all_y, dtype=torch.long)

            if isinstance(all_t[0], torch.Tensor):
                t_concat = torch.stack(all_t, dim=0) if all_t[0].ndim == 0 else torch.cat(all_t, dim=0)
            else:
                t_concat = torch.tensor(all_t, dtype=torch.long)
        else:
            X_concat = torch.empty((0, 0, 0), dtype=torch.float32)
            y_concat = torch.empty((0,), dtype=torch.long)
            t_concat = torch.empty((0,), dtype=torch.long)

        return X_concat, y_concat, t_concat

    @property
    def X(self) -> torch.Tensor:
        """Backward-compatible property returning concatenated feature tensor X."""
        if self._cached_concat_X is None:
            X_c, y_c, t_c = self._load_all_for_legacy_property()
            self._cached_concat_X = X_c
            self._cached_concat_y = y_c
            self._cached_concat_trial_ids = t_c
        return self._cached_concat_X

    @property
    def y(self) -> torch.Tensor:
        """Backward-compatible property returning concatenated label tensor y."""
        if self._cached_concat_y is None:
            X_c, y_c, t_c = self._load_all_for_legacy_property()
            self._cached_concat_X = X_c
            self._cached_concat_y = y_c
            self._cached_concat_trial_ids = t_c
        return self._cached_concat_y

    @property
    def trial_ids(self) -> torch.Tensor:
        """Backward-compatible property returning concatenated trial IDs tensor."""
        if self._cached_concat_trial_ids is None:
            X_c, y_c, t_c = self._load_all_for_legacy_property()
            self._cached_concat_X = X_c
            self._cached_concat_y = y_c
            self._cached_concat_trial_ids = t_c
        return self._cached_concat_trial_ids


class IndexingError(IndexError):
    """Custom error raised when dataset index is out of bounds."""
    pass
