"""
PyTorch HGDDataset Implementation.

Provides HGDDataset wrapping windowed motor imagery EEG signals for PyTorch
DataLoader integration.
"""

import logging
from typing import List, Union, Optional, Tuple, Dict, Any

import torch
import numpy as np
from torch.utils.data import Dataset

from datasets.pipeline import EEGPreprocessingPipeline, PreprocessingConfig

logger = logging.getLogger(__name__)


class HGDDataset(Dataset):
    """
    PyTorch Dataset wrapper for High Gamma Dataset (HGD) Motor Imagery EEG samples.

    Processes single or multiple EDF files using EEGPreprocessingPipeline,
    combining windowed samples into PyTorch tensors.
    """

    def __init__(
        self,
        file_paths: Union[str, List[str]],
        pipeline: Optional[EEGPreprocessingPipeline] = None,
        config: Optional[Union[PreprocessingConfig, str, Dict[str, Any]]] = None,
    ):
        """
        Initialize HGDDataset.

        Args:
            file_paths: Path to a single EDF file or list of EDF file paths.
            pipeline: Pre-configured EEGPreprocessingPipeline instance (optional).
            config: Configuration for pipeline if pipeline is not provided (optional).
        """
        if isinstance(file_paths, str):
            self.file_paths = [file_paths]
        else:
            self.file_paths = list(file_paths)

        if pipeline is not None:
            self.pipeline = pipeline
        else:
            self.pipeline = EEGPreprocessingPipeline(config=config)

        all_windows = []
        all_labels = []
        all_trial_ids = []

        total_trials = 0

        for f_path in self.file_paths:
            logger.info(f"HGDDataset processing file: {f_path}")
            X_win, y_win, t_ids = self.pipeline.process(f_path)

            all_windows.append(X_win)
            all_labels.append(y_win)
            # Offset trial IDs across multiple files to remain unique per trial
            all_trial_ids.append(t_ids + total_trials)

            # Count unique trials in this file
            num_file_trials = len(np.unique(t_ids)) if len(t_ids) > 0 else 0
            total_trials += num_file_trials

        if len(all_windows) > 0:
            X_concat = np.concatenate(all_windows, axis=0)
            y_concat = np.concatenate(all_labels, axis=0)
            t_concat = np.concatenate(all_trial_ids, axis=0)
        else:
            X_concat = np.empty((0, 0, 0), dtype=np.float32)
            y_concat = np.empty((0,), dtype=np.int64)
            t_concat = np.empty((0,), dtype=np.int64)

        self.X = torch.tensor(X_concat, dtype=torch.float32)
        self.y = torch.tensor(y_concat, dtype=torch.long)
        self.trial_ids = torch.tensor(t_concat, dtype=torch.long)

        self.metadata: Dict[str, Any] = {
            "num_files": len(self.file_paths),
            "num_trials": total_trials,
            "num_windows": len(self.X),
            "num_channels": self.X.shape[1] if self.X.ndim == 3 else 0,
            "window_size": self.X.shape[2] if self.X.ndim == 3 else 0,
        }

        logger.info(f"HGDDataset initialized with metadata: {self.metadata}")

    def __len__(self) -> int:
        """Return total number of cropped window samples."""
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get single sample-label pair.

        Args:
            idx (int): Sample index.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Sample tensor (shape: [n_channels, window_size]), label scalar.
        """
        return self.X[idx], self.y[idx]
