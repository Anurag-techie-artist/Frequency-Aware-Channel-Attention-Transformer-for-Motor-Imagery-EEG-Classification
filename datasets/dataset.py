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

    Uses metadata-driven index mapping (`metadata.json`) and lazy per-EDF `.pt` caching
    via FileLRUCache to keep memory footprint low (~50-100 MB RAM) during training.
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
        self.max_open_cache_files = self.cache_config.get("max_open_cache_files", 2)
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

        self._file_entries = self.metadata.get("files", [])
        self._start_indices = [e["start_index"] for e in self._file_entries]
        self._end_indices = [e["end_index"] for e in self._file_entries]
        self._total_samples = self.metadata.get("total_samples", 0)

        # Initialize LRU Cache for per-EDF sample retrieval
        self._lru_cache = FileLRUCache(max_open_cache_files=self.max_open_cache_files)

        # Lazy concatenated tensors cache for legacy compatibility properties (.X, .y)
        self._cached_concat_X: Optional[torch.Tensor] = None
        self._cached_concat_y: Optional[torch.Tensor] = None
        self._cached_concat_trial_ids: Optional[torch.Tensor] = None

        logger.info(
            f"HGDDataset initialized. Total samples: {self._total_samples}, "
            f"Cached files: {len(self._file_entries)}, Config hash: {self.config_hash}"
        )

    def __len__(self) -> int:
        """Return total number of cropped window samples."""
        return self._total_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get single sample-label pair using binary search index mapping and LRU file cache.

        Args:
            idx (int): Global sample index.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Sample tensor (shape: [Channels, Samples] or [Bands, Channels, Samples])
                - Label scalar
        """
        if idx < 0 or idx >= self._total_samples:
            raise IndexingError(f"Index {idx} out of range for dataset with {self._total_samples} samples.")

        # Fast O(log N) binary search mapping global index -> file entry
        file_idx = bisect_right(self._start_indices, idx) - 1
        file_entry = self._file_entries[file_idx]
        local_idx = idx - file_entry["start_index"]

        cache_path = os.path.join(self.cache_dir, file_entry["cache"])
        data = self._lru_cache.get(cache_path)

        return data["X"][local_idx], data["y"][local_idx]

    def _load_all_for_legacy_property(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Helper method to load and concatenate all tensors for legacy property access."""
        all_X, all_y, all_t = [], [], []
        for entry in self._file_entries:
            cache_path = os.path.join(self.cache_dir, entry["cache"])
            data = torch.load(cache_path, map_location="cpu", weights_only=False)
            all_X.append(data["X"])
            all_y.append(data["y"])
            all_t.append(data["trial_ids"])

        if len(all_X) > 0:
            X_concat = torch.cat(all_X, dim=0)
            y_concat = torch.cat(all_y, dim=0)
            t_concat = torch.cat(all_t, dim=0)
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
