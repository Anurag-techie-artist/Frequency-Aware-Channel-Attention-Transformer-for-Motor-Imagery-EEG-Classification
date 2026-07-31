"""
Test Script for EEG Preprocessing Pipeline (Phase 2).

Loads a sample EDF recording file, runs the EEGPreprocessingPipeline,
saves intermediate debug stage arrays to outputs/debug/, and logs pipeline statistics.
Phase 10 Patch v0.10.1: Centralized dataset path resolution.

Usage:
    python scripts/test_pipeline.py
"""

import os
import sys
import logging
from collections import Counter

import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.path import get_dataset_root, get_train_directory
from datasets.pipeline import EEGPreprocessingPipeline
from datasets.dataset import HGDDataset

logger = logging.getLogger(__name__)


def setup_logging():
    """Configure logging to console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def main():
    setup_logging()
    logger.info("Starting EEG Preprocessing Pipeline Test...")

    hgd_root = get_dataset_root(PROJECT_ROOT)
    train_dir = get_train_directory(PROJECT_ROOT)
    sample_edf = os.path.join(hgd_root, train_dir, "1.edf")

    if not os.path.exists(sample_edf):
        logger.error(f"Sample EDF file not found at: {sample_edf}")
        sys.exit(1)

    # 1. Initialize EEGPreprocessingPipeline
    pipeline = EEGPreprocessingPipeline()

    # 2. Process file with debug stage outputs (time domain mode)
    logger.info(f"Running pipeline debug process (time representation) on: {sample_edf}")
    debug_outputs_time = pipeline.process_debug(sample_edf, representation="time")

    raw_signal = debug_outputs_time["raw"]
    filtered_signal = debug_outputs_time["filtered"]
    epochs_data = debug_outputs_time["epochs"]
    windows_data = debug_outputs_time["windows"]
    labels_data = debug_outputs_time["labels"]

    # Save time domain intermediate debug stage arrays to outputs/debug/
    debug_dir = os.path.join(PROJECT_ROOT, "outputs", "debug")
    os.makedirs(debug_dir, exist_ok=True)

    np.save(os.path.join(debug_dir, "raw.npy"), raw_signal)
    np.save(os.path.join(debug_dir, "filtered.npy"), filtered_signal)
    np.save(os.path.join(debug_dir, "epochs.npy"), epochs_data)
    np.save(os.path.join(debug_dir, "windows.npy"), windows_data)
    np.save(os.path.join(debug_dir, "labels.npy"), labels_data)

    # 3. Process file with debug stage outputs (frequency domain mode)
    logger.info(f"Running pipeline debug process (frequency representation) on: {sample_edf}")
    debug_outputs_freq = pipeline.process_debug(sample_edf, representation="frequency")
    windows_freq_data = debug_outputs_freq["windows"]

    # 4. Demonstrate PyTorch HGDDataset integration for both modes
    logger.info("Instantiating PyTorch HGDDataset wrappers...")
    dataset_time = HGDDataset(file_paths=sample_edf, pipeline=pipeline, representation="time")
    dataset_freq = HGDDataset(file_paths=sample_edf, pipeline=pipeline, representation="frequency")

    sample_x_time, sample_y_time = dataset_time[0]
    sample_x_freq, sample_y_freq = dataset_freq[0]

    # 5. Log & Print Pipeline Statistics
    label_counts = dict(Counter(labels_data.tolist()))

    print("\n" + "=" * 60)
    print("EEG PREPROCESSING PIPELINE TEST STATISTICS")
    print("=" * 60)
    print(f"Sample EDF Path            : {sample_edf}")
    print(f"Raw Signal Shape           : {raw_signal.shape} (Channels x Time)")
    print(f"Filtered Signal Shape      : {filtered_signal.shape} (Channels x Time)")
    print(f"Extracted Epochs           : {epochs_data.shape} (Trials x Channels x Time)")
    print(f"Time Windows Shape         : {windows_data.shape} (Windows x Channels x Time)")
    print(f"Frequency Windows Shape    : {windows_freq_data.shape} (Windows x Bands x Channels x Time)")
    print(f"Window Labels Count        : {len(labels_data)} {label_counts}")
    print(f"PyTorch Time Sample        : {sample_x_time.shape}, Label: {sample_y_time.item()}")
    print(f"PyTorch Frequency Sample   : {sample_x_freq.shape}, Label: {sample_y_freq.item()}")
    print("=" * 60)
    print("Debug Outputs Saved to outputs/debug/ :")
    print(f"  - {os.path.join(debug_dir, 'raw.npy')}")
    print(f"  - {os.path.join(debug_dir, 'filtered.npy')}")
    print(f"  - {os.path.join(debug_dir, 'epochs.npy')}")
    print(f"  - {os.path.join(debug_dir, 'windows.npy')}")
    print(f"  - {os.path.join(debug_dir, 'labels.npy')}")
    print(f"  - {os.path.join(debug_dir, 'frequency_tensor.npy')}")
    print(f"  - {os.path.join(debug_dir, 'frequency_metadata.json')}")
    print(f"  - {os.path.join(debug_dir, 'frequency_summary.json')}")
    print("=" * 60)
    print("EEG Preprocessing Pipeline Test Completed Successfully!\n")


if __name__ == "__main__":
    main()
