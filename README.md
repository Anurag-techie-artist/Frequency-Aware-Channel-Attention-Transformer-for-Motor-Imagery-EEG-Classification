# Deep Learning-based Motor Imagery EEG Classification

A long-term research project for motor imagery (MI) electroencephalography (EEG) decoding using advanced deep learning architectures, frequency-aware preprocessing, attention mechanisms, and generative data augmentation.

## Overview

This repository focuses on decoding multi-class motor imagery signals from High-Gamma Dataset (HGD) EDF recordings. The goal is to incrementally develop high-performance spatio-temporal neural networks starting from baseline CNNs (EEGNet) towards Frequency-Aware Channel Attention Transformers (FA-CAT) and GAN-augmented training pipelines.

## Repository Structure

```
Deep Learning-based Motor Imagery EEG Classification/
├── hgd/                     # High-Gamma Dataset folder (EDF files)
├── basefile.py              # Original assignment starter file preserved
├── baseline/
│   └── eegnet_baseline.py   # Preserved original EEGNet implementation copy
├── datasets/
│   ├── loader.py            # Dataset loading routines & PyTorch Dataset class
│   ├── preprocessing.py     # Bandpass filtering & frequency decomposition
│   └── windowing.py         # Sliding window segmentation & trial voting
├── models/                  # Neural network model architectures
├── training/                # Training routines, loss functions, & optimizers
├── utils/                   # Metrics, logging, and plotting utilities
├── configs/                 # Experiment configuration files
├── experiments/             # Experiment execution scripts
├── scripts/                 # Utility & helper scripts
├── notebooks/               # EDA & interactive visualization notebooks
├── outputs/                 # Output artifacts (ignored in git)
│   ├── checkpoints/         # Saved model weights
│   ├── logs/                # Training logs & TensorBoard events
│   ├── plots/               # Performance plots & confusion matrices
│   ├── predictions/         # Prediction arrays & outputs
│   └── reports/             # Evaluation summary metrics
├── README.md                # Project documentation & roadmap
├── requirements.txt         # Core dependencies
├── .gitignore               # PyTorch & Python git ignore rules
├── LICENSE                  # MIT License
├── train.py                 # Training entry point script
└── test.py                  # Evaluation entry point script
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

## Planned Roadmap

1. **Phase 1: Baseline**
   - Preserve and validate baseline pipeline (`baseline/eegnet_baseline.py`).
   - Establish initial window-level accuracy and trial-level majority voting benchmarks.

2. **Phase 2: Frequency-aware Preprocessing**
   - Implement multi-band spectral decomposition (Sub-band filtering: Theta, Alpha, Beta, Gamma).
   - Dynamic channel & frequency normalization routines in `datasets/preprocessing.py`.

3. **Phase 3: Channel Attention**
   - Integrate spatial and temporal Channel Attention (CA) blocks for EEG electrode relevance weighting.

4. **Phase 4: Transformer Architecture**
   - Implement spatial-temporal Transformer encoders to capture long-range temporal dependencies in MI trials.

5. **Phase 5: GAN Data Augmentation**
   - Develop Wasserstein GAN with Gradient Penalty (WGAN-GP) for synthetic trial generation to address data scarcity.

6. **Phase 6: Final Evaluation & Benchmarking**
   - Conduct cross-subject and intra-subject benchmarking.
   - Generate comparative performance reports, confusion matrices, and ablation study plots.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run placeholder training entrypoint
python train.py

# Run placeholder evaluation entrypoint
python test.py
```

## License

This project is licensed under the [MIT License](LICENSE).
