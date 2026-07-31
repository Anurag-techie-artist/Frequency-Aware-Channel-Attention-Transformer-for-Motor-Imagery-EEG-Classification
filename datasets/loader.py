"""
EDF File Discovery, Loading, and Event Extraction Utilities.

Provides modular functions for locating HGD dataset files, reading EDF files
using MNE, extracting raw annotations, and mapping event codes.
Phase 10 Patch v0.10.1: Integrates centralized dataset path resolution.
"""

import os
import glob
import logging
from typing import List, Dict, Tuple, Optional

import numpy as np
import mne
from datasets.path import get_dataset_root

logger = logging.getLogger(__name__)


def discover_edf_files(dir_path: str) -> List[str]:
    """
    Find and sort all .edf files in a given directory.

    Args:
        dir_path (str): Path to directory containing EDF files.

    Returns:
        List[str]: Numerically sorted list of EDF file paths.
    """
    if not os.path.exists(dir_path):
        logger.warning(f"Directory path does not exist: {dir_path}")
        return []

    edf_files = glob.glob(os.path.join(dir_path, "*.edf"))

    # Sort files numerically (1.edf, 2.edf, ..., 14.edf) if possible
    sorted_files = sorted(
        edf_files,
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
        if os.path.splitext(os.path.basename(p))[0].isdigit()
        else os.path.basename(p),
    )
    return sorted_files


def locate_dataset(base_dir: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Locate train1 and test1 EDF files under the HGD dataset directory.

    Args:
        base_dir (str, optional): Base directory containing HGD splits.
                                  If None or default "./hgd", resolves via DatasetPaths.

    Returns:
        Dict[str, List[str]]: Dictionary with 'train' and 'test' file paths.
    """
    if base_dir is None or base_dir == "./hgd":
        base_dir = get_dataset_root()

    train_dir = os.path.join(base_dir, "train1")
    test_dir = os.path.join(base_dir, "test1")

    train_files = discover_edf_files(train_dir)
    test_files = discover_edf_files(test_dir)

    logger.info(f"Discovered {len(train_files)} train files in {train_dir}")
    logger.info(f"Discovered {len(test_files)} test files in {test_dir}")

    return {
        "train": train_files,
        "test": test_files,
    }


def load_raw_edf(filepath: str, preload: bool = True) -> mne.io.Raw:
    """
    Load raw EDF recording file using MNE.

    Args:
        filepath (str): Path to the EDF file.
        preload (bool): Whether to preload signal data into memory.

    Returns:
        mne.io.Raw: Loaded MNE Raw object.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"EDF file not found at: {filepath}")

    logger.info(f"Loading raw EDF file: {filepath}")
    raw = mne.io.read_raw_edf(filepath, preload=preload, verbose=False)
    return raw


def extract_annotations(raw: mne.io.Raw) -> Optional[mne.Annotations]:
    """
    Extract raw annotations from an MNE Raw object.

    Args:
        raw (mne.io.Raw): MNE Raw EDF object.

    Returns:
        Optional[mne.Annotations]: Raw annotations object or None.
    """
    return getattr(raw, "annotations", None)


def extract_events(raw: mne.io.Raw) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Extract events array and event dictionary from annotations.

    Args:
        raw (mne.io.Raw): MNE Raw EDF object.

    Returns:
        Tuple[mne.Events, Dict[str, int]]:
            - Array of events (shape: [N, 3])
            - Dictionary mapping event description string -> event ID integer
    """
    events, event_dict = mne.events_from_annotations(raw, verbose=False)
    logger.info(f"Extracted {len(events)} events with dictionary: {event_dict}")
    return events, event_dict


def validate_raw(raw: mne.io.Raw) -> bool:
    """
    Validate that an MNE Raw object is non-empty and well-formed.

    Args:
        raw (mne.io.Raw): MNE Raw EDF object.

    Returns:
        bool: True if valid.

    Raises:
        ValueError: If raw object is missing channels or samples.
    """
    if raw is None:
        raise ValueError("Raw object is None.")
    if len(raw.ch_names) == 0:
        raise ValueError("Raw EDF file contains 0 channels.")
    if raw.n_times == 0:
        raise ValueError("Raw EDF file contains 0 time samples.")
    return True
