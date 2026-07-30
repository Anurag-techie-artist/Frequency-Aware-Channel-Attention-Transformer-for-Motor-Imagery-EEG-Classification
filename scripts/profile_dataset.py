"""
HGD Dataset Profiler Script.

Main entry point to scan the High Gamma Dataset (HGD), extract metadata,
compute channel-level signal statistics, validate dataset integrity,
generate publication-quality visualizations, and export comprehensive reports.

Usage:
    python scripts/profile_dataset.py
"""

import os
import sys
import time
import logging
from typing import List, Dict, Any

# Ensure project root is in python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.metadata import (
    scan_dataset,
    compute_dataset_summary,
    generate_dataset_fingerprint,
    generate_event_dictionary,
    validate_dataset,
    export_dataset_summary_json,
    export_dataset_summary_md,
    export_dataset_fingerprint_json,
    export_event_dictionary_json,
    export_file_metadata_csv,
    export_signal_statistics_csv,
    export_raw_metadata_json,
    export_validation_report_md,
)

from utils.visualization import (
    plot_annotation_distribution,
    plot_class_distribution,
    plot_recording_duration_distribution,
    plot_channel_count_distribution,
    plot_channel_presence_heatmap,
    plot_sample_signal,
)


def setup_directories(output_base: str) -> Dict[str, str]:
    """
    Ensure output directories exist.

    Args:
        output_base (str): Path to output directory.

    Returns:
        Dict[str, str]: Paths to subdirectories.
    """
    dirs = {
        "reports": os.path.join(output_base, "reports"),
        "plots": os.path.join(output_base, "plots"),
        "logs": os.path.join(output_base, "logs"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def setup_logging(log_file_path: str) -> None:
    """
    Configure Python logging to both file and console.

    Args:
        log_file_path (str): File path for storing profiling logs.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Format
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # File Handler
    file_handler = logging.FileHandler(log_file_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Clear existing handlers
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def main() -> None:
    """Main execution function for dataset profiling."""
    t0 = time.time()

    # 1. Setup Directories
    output_dir = os.path.join(PROJECT_ROOT, "outputs")
    dir_paths = setup_directories(output_dir)

    # 2. Setup Logging
    log_file = os.path.join(dir_paths["logs"], "profiling.log")
    setup_logging(log_file)
    logger = logging.getLogger(__name__)

    logger.info("Scanning dataset...")

    hgd_path = os.path.join(PROJECT_ROOT, "hgd")
    if not os.path.exists(hgd_path):
        logger.error(f"HGD dataset directory not found at {hgd_path}!")
        sys.exit(1)

    # 3. Scan Dataset & Extract Metadata + Signal Stats
    logger.info("Extracting metadata...")
    metadata_list, signal_stats_list = scan_dataset(hgd_path)

    # 4. Compute Statistics & Fingerprint
    logger.info("Computing statistics...")
    summary = compute_dataset_summary(metadata_list)
    fingerprint = generate_dataset_fingerprint(summary)
    event_dict = generate_event_dictionary(metadata_list)
    validation_res = validate_dataset(metadata_list)

    # 5. Generate Visualizations
    logger.info("Generating plots...")
    plots_dir = dir_paths["plots"]

    plot_annotation_distribution(metadata_list, os.path.join(plots_dir, "annotation_distribution.png"))
    plot_class_distribution(metadata_list, os.path.join(plots_dir, "class_distribution.png"))
    plot_recording_duration_distribution(metadata_list, os.path.join(plots_dir, "recording_duration_distribution.png"))
    plot_channel_count_distribution(metadata_list, os.path.join(plots_dir, "channel_count_distribution.png"))
    plot_channel_presence_heatmap(metadata_list, os.path.join(plots_dir, "channel_presence_heatmap.png"))

    # Sample signals (Train and Test representative files)
    sample_train_file = os.path.join(hgd_path, "train1", "1.edf")
    sample_test_file = os.path.join(hgd_path, "test1", "1.edf")

    if os.path.exists(sample_train_file):
        plot_sample_signal(
            sample_train_file,
            title="Representative Training EEG Continuous Signals (train1/1.edf)",
            save_path=os.path.join(plots_dir, "sample_signal_train.png"),
            duration_sec=10.0,
            num_channels=5,
        )

    if os.path.exists(sample_test_file):
        plot_sample_signal(
            sample_test_file,
            title="Representative Testing EEG Continuous Signals (test1/1.edf)",
            save_path=os.path.join(plots_dir, "sample_signal_test.png"),
            duration_sec=10.0,
            num_channels=5,
        )

    # 6. Export Reports
    logger.info("Exporting reports...")
    reports_dir = dir_paths["reports"]

    export_dataset_summary_json(summary, os.path.join(reports_dir, "dataset_summary.json"))
    export_dataset_summary_md(summary, os.path.join(reports_dir, "dataset_summary.md"))
    export_dataset_fingerprint_json(fingerprint, os.path.join(reports_dir, "dataset_fingerprint.json"))
    export_event_dictionary_json(event_dict, os.path.join(reports_dir, "event_dictionary.json"))
    export_file_metadata_csv(metadata_list, os.path.join(reports_dir, "file_metadata.csv"))
    export_signal_statistics_csv(signal_stats_list, os.path.join(reports_dir, "signal_statistics.csv"))
    export_raw_metadata_json(metadata_list, os.path.join(reports_dir, "raw_metadata.json"))
    export_validation_report_md(validation_res, os.path.join(reports_dir, "validation_report.md"))

    logger.info("Dataset profiling completed.")

    elapsed_sec = time.time() - t0
    total_files = len(metadata_list)

    # Final stdout summary output
    print("\nDataset profiling completed successfully.")
    print(f"Processed {total_files} EDF files.")
    print(f"Execution time: {elapsed_sec:.2f} seconds.\n")


if __name__ == "__main__":
    main()
