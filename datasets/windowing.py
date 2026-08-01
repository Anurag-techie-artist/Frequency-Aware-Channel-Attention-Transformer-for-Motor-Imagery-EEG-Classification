"""
Sliding Window Segmentation Utilities for High Gamma Dataset (HGD).

Provides functions for cropped window generation matching baseline behavior:
- Window index computation
- Z-score normalization per trial prior to windowing
- Sliding window extraction with configurable window size, stride, and overlap
- Trial index tracking
"""

import logging
from typing import List, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


def calculate_window_indices(
    n_samples: int,
    window_size: int = 250,
    stride: int = 50
) -> List[Tuple[int, int]]:
    """
    Calculate start and end sample index pairs for sliding windows.

    Args:
        n_samples (int): Total number of time samples per trial.
        window_size (int): Length of sliding window in samples. Default: 250.
        stride (int): Step size between consecutive windows in samples. Default: 50.

    Returns:
        List[Tuple[int, int]]: List of (start_idx, end_idx) sample bounds.
    """
    indices = []
    for ws in range(0, n_samples - window_size, stride):
        we = ws + window_size
        indices.append((ws, we))
    return indices


def window_labels(y_trial: int, num_windows: int) -> np.ndarray:
    """
    Repeat trial label across all generated windows from that trial.

    Args:
        y_trial (int): Integer label for the trial.
        num_windows (int): Number of cropped windows extracted.

    Returns:
        np.ndarray: Array of shape [num_windows] filled with y_trial.
    """
    return np.full((num_windows,), y_trial, dtype=np.int64)


def generate_sliding_windows(
    X: np.ndarray,
    y: np.ndarray,
    window_size: int = 250,
    stride: int = 50,
    normalize: bool = True,
    eps: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate sliding windows from raw trial epochs matching baseline create_windows logic.

    Exact baseline workflow per trial:
    1. Apply per-trial channel Z-score normalization: (trial - mean) / (std + eps)
    2. Slice trial into overlapping windows [ws:ws+window_size] with step_size = stride
    3. Track original trial IDs for downstream majority voting

    Args:
        X (np.ndarray): Input trial array of shape [n_trials, n_channels, n_times].
        y (np.ndarray): Label array of shape [n_trials].
        window_size (int): Duration of window in samples. Default: 250.
        stride (int): Stride / step size in samples. Default: 50.
        normalize (bool): Whether to perform per-trial Z-score normalization. Default: True.
        eps (float): Epsilon for standard deviation numerical stability. Default: 1e-6.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - X_windows: Cropped windows array (shape: [N_windows, n_channels, window_size])
            - y_windows: Cropped window labels array (shape: [N_windows])
            - trial_ids: Trial index tracking array (shape: [N_windows])
    """
    X_out = []
    y_out = []
    trial_ids = []

    num_trials = len(X)
    logger.info(f"Generating sliding windows from {num_trials} trials (window_size={window_size}, stride={stride})")

    for trial_idx in range(num_trials):
        trial = X[trial_idx].copy()

        if normalize:
            mean = np.mean(trial, axis=-1, keepdims=True)
            std = np.std(trial, axis=-1, keepdims=True)
            trial = (trial - mean) / (std + eps)

        n_samples = trial.shape[-1]
        for ws in range(0, n_samples - window_size, stride):
            we = ws + window_size
            window = trial[..., ws:we]

            X_out.append(window)
            y_out.append(y[trial_idx])
            trial_ids.append(trial_idx)

    X_windows = np.array(X_out, dtype=np.float32)
    y_windows = np.array(y_out, dtype=np.int64)
    trial_ids_arr = np.array(trial_ids, dtype=np.int64)

    logger.info(
        f"Generated {len(X_windows)} cropped windows with shape {X_windows.shape}"
    )

    return X_windows, y_windows, trial_ids_arr


def extract_single_window_from_trial(
    trial: Union[np.ndarray, torch.Tensor],
    start_sample: int,
    window_size: int = 250,
    normalize: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Extract a single window from a trial array/tensor and apply per-trial Z-score normalization,
    matching exact baseline generate_sliding_windows logic.

    Args:
        trial: Input trial of shape (Channels, Times) or (Bands, Channels, Times).
        start_sample: Starting sample index within the trial.
        window_size: Window duration in samples. Default: 250.
        normalize: Whether to apply per-trial Z-score normalization. Default: True.
        eps: Epsilon for standard deviation numerical stability. Default: 1e-6.

    Returns:
        torch.Tensor: Window tensor of shape (Channels, window_size) or (Bands, Channels, window_size).
    """
    if not isinstance(trial, torch.Tensor):
        trial_t = torch.from_numpy(np.asarray(trial)).float()
    else:
        trial_t = trial.float()

    if normalize:
        mean = torch.mean(trial_t, dim=-1, keepdim=True)
        std = torch.std(trial_t, dim=-1, keepdim=True, correction=0)
        trial_t = (trial_t - mean) / (std + eps)

    end_sample = start_sample + window_size
    window = trial_t[..., start_sample:end_sample].clone()
    return window
