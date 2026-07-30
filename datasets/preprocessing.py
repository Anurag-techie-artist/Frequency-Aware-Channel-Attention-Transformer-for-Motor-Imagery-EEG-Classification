"""
Signal Preprocessing Utilities for High Gamma Dataset (HGD).

Provides modular signal manipulation routines matching baseline behavior:
- Resampling
- FIR Bandpass Filtering
- Channel Selection
- Epoch Extraction & Label Mapping
- Z-score Normalization
"""

import logging
from typing import List, Dict, Tuple, Optional

import numpy as np
import mne

from datasets.loader import load_raw_edf, extract_events

logger = logging.getLogger(__name__)


def resample_signal(raw: mne.io.Raw, target_fs: float = 250.0) -> mne.io.Raw:
    """
    Resample MNE Raw signal to target sampling frequency.

    Args:
        raw (mne.io.Raw): Input MNE Raw object.
        target_fs (float): Target sampling frequency in Hz. Default: 250.0 Hz.

    Returns:
        mne.io.Raw: Resampled MNE Raw object.
    """
    if raw.info["sfreq"] != target_fs:
        logger.info(f"Resampling raw signal from {raw.info['sfreq']} Hz to {target_fs} Hz")
        raw.resample(target_fs)
    return raw


def bandpass_filter(
    raw: mne.io.Raw,
    lowcut: float = 4.0,
    highcut: float = 38.0,
    fir_design: str = "firwin"
) -> mne.io.Raw:
    """
    Apply FIR bandpass filter to MNE Raw object.

    Args:
        raw (mne.io.Raw): Input MNE Raw object.
        lowcut (float): Lower passband edge in Hz. Default: 4.0 Hz.
        highcut (float): Upper passband edge in Hz. Default: 38.0 Hz.
        fir_design (str): FIR design method. Default: 'firwin'.

    Returns:
        mne.io.Raw: Bandpass filtered MNE Raw object.
    """
    logger.info(f"Applying bandpass filter ({lowcut} Hz - {highcut} Hz, fir_design='{fir_design}')")
    raw.filter(lowcut, highcut, fir_design=fir_design, verbose=False)
    return raw


def select_channels(raw: mne.io.Raw, channel_names: Optional[List[str]] = None) -> mne.io.Raw:
    """
    Select a subset of EEG channels.

    Args:
        raw (mne.io.Raw): Input MNE Raw object.
        channel_names (Optional[List[str]]): List of channel names to retain.

    Returns:
        mne.io.Raw: Channel-selected MNE Raw object.
    """
    if channel_names is not None:
        logger.info(f"Selecting {len(channel_names)} specified channels.")
        raw.pick_channels(channel_names)
    return raw


def normalize_signal(signal: np.ndarray, axis: int = 1, eps: float = 1e-6) -> np.ndarray:
    """
    Apply per-channel Z-score normalization (matching baseline implementation).

    Formula: (x - mean) / (std + eps)

    Args:
        signal (np.ndarray): Signal array (e.g., shape [channels, time] or [trials, channels, time]).
        axis (int): Axis along which time samples lie. Default: 1 (time axis per trial).
        eps (float): Epsilon for numerical stability. Default: 1e-6.

    Returns:
        np.ndarray: Z-score normalized signal array.
    """
    mean = np.mean(signal, axis=axis, keepdims=True)
    std = np.std(signal, axis=axis, keepdims=True)
    normalized = (signal - mean) / (std + eps)
    return normalized


def extract_epochs(
    raw: mne.io.Raw,
    events: np.ndarray,
    event_dict: Dict[str, int],
    tmin: float = 0.5,
    tmax: float = 3.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract epochs from raw signal around events and map event IDs to contiguous labels (0..N-1).

    Behavior matches baseline:
    - Label mapping sorted by event key string order
    - mne.Epochs created with baseline=None, preload=True

    Args:
        raw (mne.io.Raw): Filtered/resampled MNE Raw object.
        events (np.ndarray): Events array extracted from annotations.
        event_dict (Dict[str, int]): Dictionary of event string -> event ID.
        tmin (float): Start time relative to event in seconds. Default: 0.5s.
        tmax (float): End time relative to event in seconds. Default: 3.5s.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - X: Epochs data array (shape: [n_trials, n_channels, n_times])
            - y: Contiguous label mapped array (shape: [n_trials])
    """
    # Create label map matching baseline
    event_keys = sorted(list(event_dict.keys()))
    label_map = {event_dict[key]: idx for idx, key in enumerate(event_keys)}
    logger.info(f"Generated contiguous label map: {label_map}")

    # Epoching
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_dict,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False
    )

    X = epochs.get_data()  # shape: (n_trials, n_channels, n_times)
    y_raw = epochs.events[:, -1]
    y = np.array([label_map[code] for code in y_raw], dtype=np.int64)

    logger.info(f"Extracted epochs data shape: {X.shape}, labels shape: {y.shape}")
    return X, y


def preprocess_recording(
    filepath: str,
    fs: float = 250.0,
    lowcut: float = 4.0,
    highcut: float = 38.0,
    tmin: float = 0.5,
    tmax: float = 3.5,
    channel_names: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full functional preprocessing pipeline for a single EDF recording file.

    Exact baseline workflow:
    1. Load raw EDF
    2. Resample to fs Hz
    3. Apply FIR bandpass filter (lowcut - highcut Hz)
    4. Select channels if specified
    5. Extract annotations & events
    6. Epoch signal (tmin - tmax seconds) with contiguous label mapping

    Args:
        filepath (str): Path to EDF recording file.
        fs (float): Target sampling frequency in Hz.
        lowcut (float): Lower bandpass edge in Hz.
        highcut (float): Upper bandpass edge in Hz.
        tmin (float): Epoch start time in seconds.
        tmax (float): Epoch end time in seconds.
        channel_names (Optional[List[str]]): Channels to retain.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - X: Raw unnormalized epochs array
            - y: Label array
    """
    raw = load_raw_edf(filepath, preload=True)
    raw = resample_signal(raw, target_fs=fs)
    raw = bandpass_filter(raw, lowcut=lowcut, highcut=highcut)

    if channel_names is not None:
        raw = select_channels(raw, channel_names)

    events, event_dict = extract_events(raw)
    X, y = extract_epochs(raw, events, event_dict, tmin=tmin, tmax=tmax)

    return X, y
