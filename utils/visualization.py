"""
Publication-quality visualization utilities for High Gamma Dataset (HGD) profiling.

Generates 7 core figures:
1. annotation_distribution.png
2. class_distribution.png
3. recording_duration_distribution.png
4. channel_count_distribution.png
5. channel_presence_heatmap.png
6. sample_signal_train.png
7. sample_signal_test.png
"""

import os
import logging
from typing import List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import mne

logger = logging.getLogger(__name__)

# Use clean matplotlib style settings
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#CCCCCC"
plt.rcParams["axes.linewidth"] = 0.8


def plot_annotation_distribution(metadata_list: List[Dict[str, Any]], save_path: str) -> None:
    """
    Generate bar plot of annotation label frequencies separated by split.
    """
    logger.info(f"Generating plot: {os.path.basename(save_path)}")

    ann_train: Dict[str, int] = {}
    ann_test: Dict[str, int] = {}
    all_ann_set = set()

    for m in metadata_list:
        split = m["split"]
        for ann, cnt in m["annotation_counts"].items():
            all_ann_set.add(ann)
            if split == "train":
                ann_train[ann] = ann_train.get(ann, 0) + cnt
            else:
                ann_test[ann] = ann_test.get(ann, 0) + cnt

    all_anns = sorted(list(all_ann_set))
    train_counts = [ann_train.get(ann, 0) for ann in all_anns]
    test_counts = [ann_test.get(ann, 0) for ann in all_anns]

    x = np.arange(len(all_anns))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width / 2, train_counts, width, label="Train (`train1`)", color="#2b5c8f")
    rects2 = ax.bar(x + width / 2, test_counts, width, label="Test (`test1`)", color="#d95f02")

    ax.set_ylabel("Total Occurrences", fontsize=12, fontweight="bold")
    ax.set_title("Annotation Frequency Distribution across HGD Splits", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(all_anns, rotation=30, ha="right", fontsize=10)
    ax.legend(frameon=True, facecolor="#F8F9FA")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for rect in rects1 + rects2:
        height = rect.get_height()
        if height > 0:
            ax.annotate(f"{height}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_class_distribution(metadata_list: List[Dict[str, Any]], save_path: str) -> None:
    """
    Generate bar plot of extracted event / class label frequencies.
    """
    logger.info(f"Generating plot: {os.path.basename(save_path)}")

    event_counts_train: Dict[str, int] = {}
    event_counts_test: Dict[str, int] = {}
    all_events = set()

    for m in metadata_list:
        split = m["split"]
        for ann, cnt in m["annotation_counts"].items():
            all_events.add(ann)
            if split == "train":
                event_counts_train[ann] = event_counts_train.get(ann, 0) + cnt
            else:
                event_counts_test[ann] = event_counts_test.get(ann, 0) + cnt

    sorted_events = sorted(list(all_events))
    t_counts = [event_counts_train.get(e, 0) for e in sorted_events]
    v_counts = [event_counts_test.get(e, 0) for e in sorted_events]

    x = np.arange(len(sorted_events))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, t_counts, width, label="Train Events", color="#31a354")
    ax.bar(x + width / 2, v_counts, width, label="Test Events", color="#756bb1")

    ax.set_ylabel("Event Count", fontsize=12, fontweight="bold")
    ax.set_title("HGD Extracted Event / Class Distribution", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_events, rotation=30, ha="right", fontsize=10)
    ax.legend(frameon=True, facecolor="#F8F9FA")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_recording_duration_distribution(metadata_list: List[Dict[str, Any]], save_path: str) -> None:
    """
    Generate bar plot of recording durations (in seconds) per subject/file.
    """
    logger.info(f"Generating plot: {os.path.basename(save_path)}")

    train_meta = [m for m in metadata_list if m["split"] == "train"]
    test_meta = [m for m in metadata_list if m["split"] == "test"]

    fig, ax = plt.subplots(figsize=(12, 6))

    files_t = [m["filename"].replace(".edf", "") for m in train_meta]
    durations_t = [m["recording_duration"] for m in train_meta]

    files_v = [m["filename"].replace(".edf", "") for m in test_meta]
    durations_v = [m["recording_duration"] for m in test_meta]

    x_t = np.arange(len(files_t))
    x_v = np.arange(len(files_v))

    ax.plot(x_t, durations_t, marker="o", color="#2b5c8f", linewidth=2, label="Train Recording Duration (s)")
    ax.plot(x_v, durations_v, marker="s", color="#d95f02", linewidth=2, linestyle="--", label="Test Recording Duration (s)")

    ax.set_xlabel("Subject ID / File Index", fontsize=12, fontweight="bold")
    ax.set_ylabel("Duration (seconds)", fontsize=12, fontweight="bold")
    ax.set_title("HGD Recording Duration Profile per Subject File", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x_t)
    ax.set_xticklabels(files_t, rotation=0, fontsize=10)
    ax.legend(frameon=True, facecolor="#F8F9FA")
    ax.grid(True, linestyle="--", alpha=0.5)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_channel_count_distribution(metadata_list: List[Dict[str, Any]], save_path: str) -> None:
    """
    Generate bar plot of channel counts per file.
    """
    logger.info(f"Generating plot: {os.path.basename(save_path)}")

    file_labels = [f"{m['split'][0].upper()}:{m['filename'].replace('.edf', '')}" for m in metadata_list]
    counts = [m["number_of_channels"] for m in metadata_list]

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["#2b5c8f" if m["split"] == "train" else "#d95f02" for m in metadata_list]

    ax.bar(file_labels, counts, color=colors, width=0.6)
    ax.set_xlabel("EDF File (T=Train, T=Test)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Channels", fontsize=12, fontweight="bold")
    ax.set_title("EEG Channel Count per EDF File across HGD Dataset", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylim(0, max(counts) + 10)
    ax.set_xticklabels(file_labels, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for i, v in enumerate(counts):
        ax.text(i, v + 0.8, str(v), ha="center", va="bottom", fontsize=8)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_channel_presence_heatmap(metadata_list: List[Dict[str, Any]], save_path: str) -> None:
    """
    Generate heatmap showing channel presence (Present=1, Missing=0) across all EDF files.
    """
    logger.info(f"Generating plot: {os.path.basename(save_path)}")

    # Collect all unique channel names sorted
    all_channels = sorted(list(set(ch for m in metadata_list for ch in m["channel_names"])))

    row_labels = [f"{m['split']}/{m['filename']}" for m in metadata_list]
    matrix = np.zeros((len(metadata_list), len(all_channels)), dtype=int)

    for r_idx, m in enumerate(metadata_list):
        file_chans = set(m["channel_names"])
        for c_idx, ch in enumerate(all_channels):
            if ch in file_chans:
                matrix[r_idx, c_idx] = 1

    fig, ax = plt.subplots(figsize=(16, 10))
    cax = ax.matshow(matrix, cmap=plt.cm.colors.ListedColormap(["#f0f0f0", "#2b5c8f"]), aspect="auto")

    ax.set_xticks(np.arange(len(all_channels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(all_channels, rotation=90, fontsize=7)
    ax.set_yticklabels(row_labels, fontsize=8)

    ax.set_title("HGD Channel Presence Matrix across EDF Files", fontsize=14, fontweight="bold", pad=25)
    ax.set_xlabel("EEG Channels", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("EDF Files", fontsize=11, fontweight="bold")

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2b5c8f", edgecolor="black", label="Present (1)"),
        Patch(facecolor="#f0f0f0", edgecolor="black", label="Missing (0)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.15, 1.05))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_sample_signal(
    edf_path: str,
    title: str,
    save_path: str,
    duration_sec: float = 10.0,
    num_channels: int = 5
) -> None:
    """
    Plot representative continuous EEG signal traces (first num_channels, first duration_sec).
    """
    logger.info(f"Generating signal preview plot: {os.path.basename(save_path)}")

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    sfreq = float(raw.info["sfreq"])
    n_samples = int(duration_sec * sfreq)

    data, times = raw[:num_channels, :n_samples]
    ch_names = raw.ch_names[:num_channels]

    fig, axes = plt.subplots(num_channels, 1, figsize=(12, 8), sharex=True)
    if num_channels == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.95)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for i in range(num_channels):
        color = colors[i % len(colors)]
        axes[i].plot(times, data[i] * 1e6, color=color, linewidth=1.0)  # Scale to microvolts (µV)
        axes[i].set_ylabel(f"{ch_names[i]}\n(µV)", fontsize=9, rotation=0, labelpad=25, va="center")
        axes[i].grid(True, linestyle="--", alpha=0.5)
        axes[i].tick_params(labelsize=8)

    axes[-1].set_xlabel("Time (seconds)", fontsize=11, fontweight="bold")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=300)
    plt.close()
