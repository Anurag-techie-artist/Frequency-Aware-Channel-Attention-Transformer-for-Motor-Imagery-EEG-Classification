# Deep Learning-based Motor Imagery EEG Classification

A long-term research project for motor imagery (MI) electroencephalography (EEG) decoding using advanced deep learning architectures, frequency-aware preprocessing, attention mechanisms, and generative data augmentation.

## Overview

This repository focuses on decoding multi-class motor imagery signals from High-Gamma Dataset (HGD) EDF recordings. The goal is to incrementally develop high-performance spatio-temporal neural networks starting from baseline CNNs (EEGNet) towards Frequency-Aware Channel Attention Transformers (FA-CAT) and GAN-augmented training pipelines.

---

## Dataset Configuration (`v0.10.1` Patch)

The dataset location is centrally resolved in **exactly one place** (`datasets/path.py`), eliminating all hardcoded paths across OS platforms (Windows, Linux, WSL, macOS) while preserving **100% backward compatibility**.

### Resolution Priority Hierarchy

```text
1. Environment Variable: HGD_DATASET_ROOT
2. configs/dataset.yaml   (dataset.root)
3. Default Fallback:       ./hgd
```

### Setup Options

#### Option 1 (Recommended): Environment Variable

- **Linux / macOS**:
  ```bash
  export HGD_DATASET_ROOT=/path/to/hgd
  ```
- **WSL**:
  ```bash
  export HGD_DATASET_ROOT=/mnt/c/Datasets/HGD
  ```
- **Windows PowerShell**:
  ```powershell
  $env:HGD_DATASET_ROOT="D:\Datasets\HGD"
  ```

#### Option 2: Configuration File (`configs/dataset.yaml`)

Edit `configs/dataset.yaml`:
```yaml
dataset:
  root: "/path/to/hgd"
  train_directory: "train1"
  test_directory: "test1"
  description: "High-Gamma Dataset"
```

#### Option 3: Default Repository Layout

Place the `hgd/` folder directly inside the repository root (`./hgd`). No setup required!

---

## Repository Structure

```
Deep Learning-based Motor Imagery EEG Classification/
├── hgd/                     # High-Gamma Dataset folder (EDF files)
├── basefile.py              # Original assignment starter file preserved
├── baseline/
│   └── eegnet_baseline.py   # Preserved original EEGNet implementation copy
├── datasets/
│   ├── path.py              # Centralized dataset path resolution & validation
│   ├── loader.py            # Dataset loading routines & PyTorch Dataset class
│   ├── preprocessing.py     # Bandpass filtering & frequency decomposition
│   └── windowing.py         # Sliding window segmentation & trial voting
├── models/                  # Neural network model architectures
├── training/                # Training routines, loss functions, & optimizers
├── utils/                   # Metrics, logging, and plotting utilities
├── configs/                 # Experiment configuration files
│   ├── dataset.yaml         # Centralized dataset location configuration
│   ├── preprocessing.yaml   # Preprocessing configuration
│   ├── model.yaml           # Model architecture configuration
│   ├── train.yaml           # Training configuration
│   ├── hpo.yaml             # Hyperparameter optimization configuration
│   └── gan.yaml             # WGAN-GP data augmentation configuration
├── augmentation/            # EEG data augmentation framework
├── experiments/             # Experiment execution scripts
├── scripts/                 # Utility & helper scripts
├── notebooks/               # EDA & interactive visualization notebooks
├── outputs/                 # Output artifacts (ignored in git)
├── README.md                # Project documentation & roadmap
├── requirements.txt         # Core dependencies
└── LICENSE                  # MIT License
```

## Dataset Profiling

The repository includes an automated, non-destructive dataset profiler pipeline to inspect, validate, and extract statistics from High Gamma Dataset (HGD) EDF files (`hgd/train1/` and `hgd/test1/`).

### Purpose of Dataset Profiling

Dataset profiling establishes a rigorous data foundation prior to preprocessing and model development. It automatically:
- Validates EDF file readability, channel counts, and sampling rate consistency.
- Computes channel-level signal statistics (min, max, mean, standard deviation, RMS) to detect noisy/dead electrodes or scaling anomalies.
- Generates reproducible dataset fingerprints and event dictionaries used by future pipeline phases.
- Produces publication-quality visualizations for dataset documentation and exploratory data analysis (EDA).

### Running the Profiler

```bash
python scripts/profile_dataset.py
```

### Generated Reports & Visualizations

Running the profiler automatically creates structured outputs under `outputs/`:

```
outputs/
├── logs/
│   └── profiling.log                      # Complete execution logs
├── plots/
│   ├── annotation_distribution.png        # Annotation label frequency breakdown
│   ├── class_distribution.png             # Extracted event/class distribution
│   ├── recording_duration_distribution.png # Duration (seconds) per EDF recording
│   ├── channel_count_distribution.png     # EEG channel count per file
│   ├── channel_presence_heatmap.png       # Electrode presence matrix (Files x Channels)
│   ├── sample_signal_train.png            # Continuous EEG signal trace (train split)
│   └── sample_signal_test.png             # Continuous EEG signal trace (test split)
└── reports/
    ├── dataset_summary.json               # Aggregated JSON dataset summary
    ├── dataset_summary.md                 # Formatted Markdown report & file table
    ├── dataset_fingerprint.json           # Reproducible dataset version fingerprint
    ├── event_dictionary.json              # Discovered annotation-to-event mappings
    ├── file_metadata.csv                  # File-wise metadata table
    ├── signal_statistics.csv              # Per-channel signal statistics (min, max, mean, std, RMS)
    ├── raw_metadata.json                  # Complete unaggregated metadata dump
    └── validation_report.md               # Automated dataset integrity validation report
```

## Preprocessing Pipeline

The `datasets/` package provides a modular, extensible, and configurable data pipeline that extracts preprocessing logic from the baseline implementation while preserving 100% functional equivalence.

### Architecture

```
datasets/
├── path.py                   # Centralized dataset path resolution & validation
├── loader.py                 # EDF file discovery, raw MNE loading, & event extraction
├── preprocessing.py          # Resampling, FIR bandpass filtering, & Z-score normalization
├── windowing.py              # Sliding window segmentation & trial index tracking
├── pipeline.py               # EEGPreprocessingPipeline orchestrator class
└── dataset.py                # PyTorch HGDDataset wrapper for DataLoader integration
```

### EEGPreprocessingPipeline

The `EEGPreprocessingPipeline` encapsulates sequential processing stages:
`load_raw` → `resample` → `filter` → `epoch` → `normalize` → `window`

```python
from datasets import EEGPreprocessingPipeline, HGDDataset
from datasets.path import get_dataset_root, get_train_directory

hgd_root = get_dataset_root()
train_dir = get_train_directory()
sample_file = f"{hgd_root}/{train_dir}/1.edf"

# Process a single EDF recording file
pipeline = EEGPreprocessingPipeline(config="configs/preprocessing.yaml")
X_windows, y_windows, trial_ids = pipeline.process(sample_file)

# Wrap in PyTorch Dataset
dataset = HGDDataset(file_paths=sample_file, pipeline=pipeline)
```

### Testing & Equivalence Verification

Run the test script to process a recording and save intermediate debug stage arrays (`raw.npy`, `filtered.npy`, `epochs.npy`, `windows.npy`, `labels.npy`) under `outputs/debug/`:

```bash
python scripts/test_pipeline.py
```

Run unit tests verifying 100% numerical match against baseline functions:

```bash
python -m unittest tests/test_pipeline_equivalence.py
```

## Unit Test Verification

Run all test suites across dataset path resolution, pipeline equivalence, model architecture, training state, evaluation metrics, HPO framework, and WGAN-GP data augmentation:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## License

This project is licensed under the [MIT License](LICENSE).
