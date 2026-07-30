"""
Test Script for EEG Preprocessing Pipeline (Phase 2).

Loads a sample EDF recording file, runs the EEGPreprocessingPipeline,
saves intermediate debug stage arrays to outputs/debug/, and logs pipeline statistics.

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

    sample_edf = os.path.join(PROJECT_ROOT, "hgd", "train1", "1.edf")
    if not os.path.exists(sample_edf):
        logger.error(f"Sample EDF file not found at: {sample_edf}")
        sys.exit(1)

    # 1. Initialize EEGPreprocessingPipeline
    pipeline = EEGPreprocessingPipeline()

    # 2. Process file with debug stage outputs
    logger.info(f"Running pipeline debug process on: {sample_edf}")
    debug_outputs = pipeline.process_debug(sample_edf)

    raw_signal = debug_outputs["raw"]
    filtered_signal = debug_outputs["filtered"]
    epochs_data = debug_outputs["epochs"]
    windows_data = debug_outputs["windows"]
    labels_data = debug_outputs["labels"]
    trial_ids = debug_outputs["trial_ids"]

    # 3. Save intermediate debug stage arrays to outputs/debug/
    debug_dir = os.path.join(PROJECT_ROOT, "outputs", "debug")
    os.makedirs(debug_dir, exist_ok=True)

    np.save(os.path.join(debug_dir, "raw.npy"), raw_signal)
    np.save(os.path.join(debug_dir, "filtered.npy"), filtered_signal)
    np.save(os.path.join(debug_dir, "epochs.npy"), epochs_data)
    np.save(os.path.join(debug_dir, "windows.npy"), windows_data)
    np.save(os.path.join(debug_dir, "labels.npy"), labels_data)

    logger.info(f"Saved intermediate debug arrays to: {debug_dir}")

    # 4. Demonstrate PyTorch HGDDataset integration
    logger.info("Instantiating PyTorch HGDDataset wrapper...")
    dataset = HGDDataset(file_paths=sample_edf, pipeline=pipeline)

    sample_x, sample_y = dataset[0]

    # 5. Log & Print Pipeline Statistics
    label_counts = dict(Counter(labels_data.tolist()))

    print("\n" + "=" * 60)
    print("EEG PREPROCESSING PIPELINE TEST STATISTICS")
    print("=" * 60)
    print(f"Sample EDF Path      : {sample_edf}")
    print(f"Raw Signal Shape     : {raw_signal.shape} (Channels x Time)")
    print(f"Filtered Signal Shape: {filtered_signal.shape} (Channels x Time)")
    print(f"Extracted Epochs     : {epochs_data.shape} (Trials x Channels x Time)")
    print(f"Generated Windows    : {windows_data.shape} (Windows x Channels x Time)")
    print(f"Window Labels Count  : {len(labels_data)} {label_counts}")
    print(f"PyTorch Sample Tensor: {sample_x.shape}, Label: {sample_y.item()}")
    print("=" * 60)
    print("Debug Outputs Saved  :")
    print(f"  - {os.path.join(debug_dir, 'raw.npy')}")
    print(f"  - {os.path.join(debug_dir, 'filtered.npy')}")
    print(f"  - {os.path.join(debug_dir, 'epochs.npy')}")
    print(f"  - {os.path.join(debug_dir, 'windows.npy')}")
    print(f"  - {os.path.join(debug_dir, 'labels.npy')}")
    print("=" * 60)
    print("EEG Preprocessing Pipeline Test Completed Successfully!\n")


if __name__ == "__main__":
    main()
