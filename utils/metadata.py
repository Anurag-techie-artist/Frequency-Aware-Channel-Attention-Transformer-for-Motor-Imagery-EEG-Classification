"""
Metadata extraction and dataset profiling utilities for High Gamma Dataset (HGD).

This module handles:
- EDF file metadata extraction using MNE
- Per-channel signal statistics (min, max, mean, std, RMS)
- Dataset-wide aggregation and validation
- Exporting reports in JSON, Markdown, CSV formats
"""

import os
import glob
import json
import csv
import logging
from datetime import datetime
from collections import Counter
from typing import List, Dict, Tuple, Any

import numpy as np
import mne
from tqdm import tqdm

logger = logging.getLogger(__name__)


def extract_file_metadata(filepath: str, split: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Extract metadata and signal statistics from a single EDF file.

    Args:
        filepath (str): Absolute or relative path to the EDF file.
        split (str): Dataset split ('train' or 'test').

    Returns:
        Tuple[Dict[str, Any], List[Dict[str, Any]]]:
            - File metadata dictionary
            - List of channel signal statistics dictionaries
    """
    rel_filename = os.path.basename(filepath)
    logger.info(f"Reading EDF: {filepath}")

    # Read raw EDF file using MNE
    raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)

    sfreq = float(raw.info["sfreq"])
    n_times = raw.n_times
    duration = float(n_times / sfreq) if sfreq > 0 else 0.0
    channel_names = [str(ch) for ch in raw.ch_names]
    n_channels = len(channel_names)

    meas_date = raw.info.get("meas_date")
    start_time_str = meas_date.isoformat() if meas_date is not None else "N/A"

    # Annotations
    annotations = raw.annotations
    num_annotations = len(annotations) if annotations is not None else 0
    annotation_counts: Dict[str, int] = {}
    if num_annotations > 0:
        desc_list = [str(desc) for desc in annotations.description]
        annotation_counts = dict(Counter(desc_list))
    annotation_labels = sorted(list(annotation_counts.keys()))

    # Events
    events_list = []
    event_dict: Dict[str, int] = {}
    try:
        events_arr, raw_event_dict = mne.events_from_annotations(raw, verbose=False)
        num_events = len(events_arr)
        event_dict = {str(k): int(v) for k, v in raw_event_dict.items()}
    except Exception as e:
        logger.warning(f"Could not extract events from annotations for {rel_filename}: {e}")
        num_events = 0

    file_meta = {
        "filename": rel_filename,
        "filepath": filepath,
        "split": split,
        "recording_duration": round(duration, 2),
        "sampling_frequency": sfreq,
        "number_of_channels": n_channels,
        "channel_names": channel_names,
        "recording_start_time": start_time_str,
        "number_of_annotations": num_annotations,
        "annotation_labels": annotation_labels,
        "annotation_counts": annotation_counts,
        "event_dictionary": event_dict,
        "number_of_events": num_events,
    }

    # Signal Statistics per channel
    signal_data = raw.get_data()  # shape: (n_channels, n_times)
    channel_stats: List[Dict[str, Any]] = []

    for ch_idx, ch_name in enumerate(channel_names):
        ch_signal = signal_data[ch_idx]
        ch_min = float(np.min(ch_signal))
        ch_max = float(np.max(ch_signal))
        ch_mean = float(np.mean(ch_signal))
        ch_std = float(np.std(ch_signal))
        ch_rms = float(np.sqrt(np.mean(ch_signal ** 2)))

        channel_stats.append({
            "filename": rel_filename,
            "split": split,
            "channel": ch_name,
            "min": round(ch_min, 6),
            "max": round(ch_max, 6),
            "mean": round(ch_mean, 6),
            "std": round(ch_std, 6),
            "rms": round(ch_rms, 6),
        })

    return file_meta, channel_stats


def scan_dataset(hgd_dir: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Scan all EDF files under train1 and test1 subdirectories of HGD.

    Args:
        hgd_dir (str): Base directory path for HGD dataset.

    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
            - List of file metadata entries
            - List of channel signal statistics entries
    """
    logger.info(f"Scanning dataset under base directory: {hgd_dir}")

    subdirs = [("train1", "train"), ("test1", "test")]
    file_queue: List[Tuple[str, str]] = []

    for sub_folder, split_name in subdirs:
        dir_path = os.path.join(hgd_dir, sub_folder)
        if os.path.exists(dir_path):
            edf_paths = glob.glob(os.path.join(dir_path, "*.edf"))
            # Sort numerical filenames correctly (1.edf, 2.edf, ... 14.edf)
            edf_paths = sorted(
                edf_paths,
                key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
                if os.path.splitext(os.path.basename(p))[0].isdigit()
                else os.path.basename(p)
            )
            for p in edf_paths:
                file_queue.append((p, split_name))
        else:
            logger.warning(f"Subdirectory {dir_path} does not exist!")

    metadata_list: List[Dict[str, Any]] = []
    signal_stats_list: List[Dict[str, Any]] = []

    logger.info(f"Found {len(file_queue)} EDF files to profile.")

    for filepath, split in tqdm(file_queue, desc="Profiling EDF Files", unit="file"):
        logger.info(f"Extracting metadata from {os.path.basename(filepath)}")
        file_meta, stats = extract_file_metadata(filepath, split)
        metadata_list.append(file_meta)
        signal_stats_list.extend(stats)

    return metadata_list, signal_stats_list


def compute_dataset_summary(metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate file metadata into high-level dataset statistics.

    Args:
        metadata_list (List[Dict[str, Any]]): Extracted metadata per file.

    Returns:
        Dict[str, Any]: Consolidated dataset summary dictionary.
    """
    logger.info("Computing summary statistics across all scanned files...")

    total_files = len(metadata_list)
    train_count = sum(1 for m in metadata_list if m["split"] == "train")
    test_count = sum(1 for m in metadata_list if m["split"] == "test")

    durations = [m["recording_duration"] for m in metadata_list]
    channels_counts = [m["number_of_channels"] for m in metadata_list]
    sfreqs = sorted(list(set(m["sampling_frequency"] for m in metadata_list)))

    all_channels = set()
    for m in metadata_list:
        all_channels.update(m["channel_names"])
    unique_channels = sorted(list(all_channels))

    all_annotations = set()
    global_annotation_counts: Dict[str, int] = {}
    for m in metadata_list:
        for ann, cnt in m["annotation_counts"].items():
            all_annotations.add(ann)
            global_annotation_counts[ann] = global_annotation_counts.get(ann, 0) + cnt
    unique_annotation_labels = sorted(list(all_annotations))

    total_events = sum(m["number_of_events"] for m in metadata_list)

    summary = {
        "total_edf_files": total_files,
        "train_count": train_count,
        "test_count": test_count,
        "average_recording_duration_sec": round(float(np.mean(durations)), 2) if durations else 0.0,
        "min_recording_duration_sec": round(float(np.min(durations)), 2) if durations else 0.0,
        "max_recording_duration_sec": round(float(np.max(durations)), 2) if durations else 0.0,
        "average_channels": round(float(np.mean(channels_counts)), 2) if channels_counts else 0.0,
        "unique_channels": unique_channels,
        "number_of_unique_channels": len(unique_channels),
        "unique_annotation_labels": unique_annotation_labels,
        "sampling_frequencies_hz": sfreqs,
        "event_statistics": {
            "total_events": total_events,
            "global_annotation_counts": global_annotation_counts,
        },
        "file_wise_metadata": metadata_list,
    }
    return summary


def generate_dataset_fingerprint(summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a reproducible dataset fingerprint dictionary.

    Args:
        summary (Dict[str, Any]): Dataset summary dictionary.

    Returns:
        Dict[str, Any]: Dataset fingerprint.
    """
    fingerprint = {
        "dataset": "HGD",
        "edf_files": summary["total_edf_files"],
        "train_files": summary["train_count"],
        "test_files": summary["test_count"],
        "sampling_rates": summary["sampling_frequencies_hz"],
        "unique_channels": summary["unique_channels"],
        "number_of_unique_channels": summary["number_of_unique_channels"],
        "unique_annotations": summary["unique_annotation_labels"],
        "total_events": summary["event_statistics"]["total_events"],
        "generated_on": datetime.now().isoformat(),
        "version": "v1",
    }
    return fingerprint


def generate_event_dictionary(metadata_list: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Consolidate all annotation/event label mappings discovered across files.

    Args:
        metadata_list (List[Dict[str, Any]]): List of file metadata entries.

    Returns:
        Dict[str, int]: Event dictionary mapping label -> event_id.
    """
    event_dict: Dict[str, int] = {}
    for m in metadata_list:
        for label, code in m["event_dictionary"].items():
            if label not in event_dict:
                event_dict[label] = code
    return dict(sorted(event_dict.items(), key=lambda item: item[1]))


def validate_dataset(metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate dataset consistency and integrity.

    Args:
        metadata_list (List[Dict[str, Any]]): List of file metadata entries.

    Returns:
        Dict[str, Any]: Structured validation report dictionary.
    """
    logger.info("Executing dataset validation checks...")

    checks = []
    status_ok = True

    # Check 1: File readability
    num_files = len(metadata_list)
    checks.append({
        "check": "EDF File Readability",
        "status": "PASSED" if num_files > 0 else "FAILED",
        "details": f"Successfully parsed {num_files} EDF files using MNE."
    })

    # Check 2: Sampling frequency consistency
    sfreqs = set(m["sampling_frequency"] for m in metadata_list)
    sfreq_pass = len(sfreqs) == 1
    checks.append({
        "check": "Sampling Frequency Consistency",
        "status": "PASSED" if sfreq_pass else "WARNING",
        "details": f"Discovered sampling rates (Hz): {sorted(list(sfreqs))}"
    })

    # Check 3: Channel count consistency
    channel_counts = set(m["number_of_channels"] for m in metadata_list)
    ch_pass = len(channel_counts) == 1
    checks.append({
        "check": "Channel Count Uniformity",
        "status": "PASSED" if ch_pass else "WARNING",
        "details": f"Channel count variations across files: {sorted(list(channel_counts))}"
    })

    # Check 4: Duplicate Filenames
    filenames = [m["filename"] + "_" + m["split"] for m in metadata_list]
    dup_pass = len(filenames) == len(set(filenames))
    checks.append({
        "check": "Filename Uniqueness",
        "status": "PASSED" if dup_pass else "FAILED",
        "details": "All filename-split pairs are unique." if dup_pass else "Duplicate filename detected!"
    })

    # Check 5: Annotations Presence
    files_without_ann = [m["filename"] for m in metadata_list if m["number_of_annotations"] == 0]
    ann_pass = len(files_without_ann) == 0
    checks.append({
        "check": "Annotation Presence",
        "status": "PASSED" if ann_pass else "WARNING",
        "details": "All files contain annotations." if ann_pass else f"Files missing annotations: {files_without_ann}"
    })

    # Check 6: Event Extraction
    files_without_events = [m["filename"] for m in metadata_list if m["number_of_events"] == 0]
    event_pass = len(files_without_events) == 0
    checks.append({
        "check": "Event Extraction",
        "status": "PASSED" if event_pass else "WARNING",
        "details": "Events extracted successfully from all files." if event_pass else f"Files with 0 events: {files_without_events}"
    })

    return {
        "overall_status": "PASSED" if status_ok else "FAILED",
        "timestamp": datetime.now().isoformat(),
        "total_files_checked": num_files,
        "checks": checks,
    }


# ============================================================
# EXPORTERS
# ============================================================

def export_dataset_summary_json(summary: Dict[str, Any], path: str) -> None:
    """Export summary dictionary as formatted JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    logger.info(f"Exported dataset summary JSON to {path}")


def export_dataset_summary_md(summary: Dict[str, Any], path: str) -> None:
    """Export dataset summary as a structured Markdown document."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    md = []
    md.append("# High Gamma Dataset (HGD) Profiling Summary\n")
    md.append(f"**Generated On:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
    md.append("## Executive Metrics\n")
    md.append(f"- **Total EDF Files:** {summary['total_edf_files']}")
    md.append(f"- **Train Files (`train1`):** {summary['train_count']}")
    md.append(f"- **Test Files (`test1`):** {summary['test_count']}")
    md.append(f"- **Sampling Frequencies (Hz):** {summary['sampling_frequencies_hz']}")
    md.append(f"- **Average Duration (s):** {summary['average_recording_duration_sec']} (Range: {summary['min_recording_duration_sec']}s - {summary['max_recording_duration_sec']}s)")
    md.append(f"- **Average Channels:** {summary['average_channels']}")
    md.append(f"- **Unique EEG Channels:** {summary['number_of_unique_channels']}")
    md.append(f"- **Total Extracted Events:** {summary['event_statistics']['total_events']}\n")

    md.append("## Annotation Frequencies\n")
    md.append("| Annotation Label | Total Occurrences |")
    md.append("| :--- | :--- |")
    for ann, cnt in summary["event_statistics"]["global_annotation_counts"].items():
        md.append(f"| `{ann}` | {cnt} |")
    md.append("")

    md.append("## File-wise Metadata Table\n")
    md.append("| Filename | Split | Duration (s) | Sampling Rate (Hz) | Channels | Annotations | Events | Start Time |")
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for m in summary["file_wise_metadata"]:
        md.append(
            f"| `{m['filename']}` | `{m['split']}` | {m['recording_duration']} | "
            f"{m['sampling_frequency']} | {m['number_of_channels']} | "
            f"{m['number_of_annotations']} | {m['number_of_events']} | {m['recording_start_time']} |"
        )
    md.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    logger.info(f"Exported dataset summary Markdown to {path}")


def export_dataset_fingerprint_json(fingerprint: Dict[str, Any], path: str) -> None:
    """Export dataset fingerprint as JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fingerprint, f, indent=4)
    logger.info(f"Exported dataset fingerprint to {path}")


def export_event_dictionary_json(event_dict: Dict[str, int], path: str) -> None:
    """Export event dictionary mapping as JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(event_dict, f, indent=4)
    logger.info(f"Exported event dictionary to {path}")


def export_file_metadata_csv(metadata_list: List[Dict[str, Any]], path: str) -> None:
    """Export file-wise metadata as a CSV file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "filename", "split", "recording_duration", "sampling_frequency",
        "number_of_channels", "number_of_annotations", "number_of_events",
        "recording_start_time"
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for m in metadata_list:
            writer.writerow(m)
    logger.info(f"Exported file metadata CSV to {path}")


def export_signal_statistics_csv(signal_stats_list: List[Dict[str, Any]], path: str) -> None:
    """Export channel-level signal statistics as a CSV file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["filename", "split", "channel", "min", "max", "mean", "std", "rms"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(signal_stats_list)
    logger.info(f"Exported signal statistics CSV to {path}")


def export_raw_metadata_json(metadata_list: List[Dict[str, Any]], path: str) -> None:
    """Export complete raw metadata list as JSON for downstream reusability."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=4)
    logger.info(f"Exported raw metadata JSON to {path}")


def export_validation_report_md(validation_res: Dict[str, Any], path: str) -> None:
    """Export validation results as a clean Markdown report."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    md = []
    md.append("# High Gamma Dataset (HGD) Validation Report\n")
    md.append(f"**Timestamp:** {validation_res['timestamp']}  ")
    md.append(f"**Overall Validation Status:** `{validation_res['overall_status']}`  ")
    md.append(f"**Total Files Checked:** {validation_res['total_files_checked']}\n")

    md.append("## Automated Validation Checks\n")
    md.append("| Check | Status | Details |")
    md.append("| :--- | :---: | :--- |")

    for c in validation_res["checks"]:
        status_icon = "✓" if c["status"] == "PASSED" else ("⚠" if c["status"] == "WARNING" else "❌")
        md.append(f"| {c['check']} | {status_icon} **{c['status']}** | {c['details']} |")

    md.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    logger.info(f"Exported validation report Markdown to {path}")
