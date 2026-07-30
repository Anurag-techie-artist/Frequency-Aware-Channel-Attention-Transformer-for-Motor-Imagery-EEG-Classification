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

## Preprocessing Pipeline

The `datasets/` package provides a modular, extensible, and configurable data pipeline that extracts preprocessing logic from the baseline implementation while preserving 100% functional equivalence.

### Architecture

```
datasets/
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

# Process a single EDF recording file
pipeline = EEGPreprocessingPipeline(config="configs/preprocessing.yaml")
X_windows, y_windows, trial_ids = pipeline.process("hgd/train1/1.edf")

# Wrap in PyTorch Dataset
dataset = HGDDataset(file_paths="hgd/train1/1.edf", pipeline=pipeline)
```

### Configuration (`configs/preprocessing.yaml`)

Initialized with baseline default reference parameters (configurable for future experiments):
- `sampling_rate`: `250` Hz
- `filter_low`: `4.0` Hz
- `filter_high`: `38.0` Hz
- `epoch_start`: `0.5` s
- `epoch_end`: `3.5` s
- `window_size`: `250` samples
- `window_stride`: `50` samples
- `normalization`: `"zscore"`

### Testing & Equivalence Verification

Run the test script to process a recording and save intermediate debug stage arrays (`raw.npy`, `filtered.npy`, `epochs.npy`, `windows.npy`, `labels.npy`) under `outputs/debug/`:

```bash
python scripts/test_pipeline.py
```

Run unit tests verifying 100% numerical match against baseline functions:

```bash
python -m unittest tests/test_pipeline_equivalence.py
```

## Frequency-Aware EEG Representation (Phase 3)

The `datasets/transforms/` package introduces modular signal transformations. The `FrequencyRepresentation` class decomposes EEG signals into multi-band spectral representations using zero-phase FIR filtering via MNE (`mne.filter.filter_data`).

### Motivation & Neurophysiological Rationale

Motor Imagery (MI) tasks induce neurophysiological phenomena known as Event-Related Desynchronization (ERD) and Event-Related Synchronization (ERS). These phenomena manifest in distinct frequency bands across the sensorimotor cortex:
- **Theta ($\mathbf{4\text{--}8\text{ Hz}}$)**: Frontal midline synchronization during task initiation and cognitive processing.
- **Alpha ($\mathbf{8\text{--}13\text{ Hz}}$)**: Mu rhythm ERD over sensorimotor cortex during imagery execution.
- **Beta ($\mathbf{13\text{--}30\text{ Hz}}$)**: ERD during motor execution/imagery and post-imagery ERS (beta rebound).
- **Gamma ($\mathbf{30\text{--}38\text{ Hz}}$)**: High-frequency local network synchronization and fine motor control representation.

By decomposing standard time-domain signals into multi-band spectral tensors, downstream attention and transformer architectures can dynamically weight both spatial electrode contributions and frequency band relevance.

### FIR Zero-Phase Filtering

To eliminate phase distortion and preserve temporal alignment across frequency bands:
- **Filter Method**: Finite Impulse Response (FIR) filter design using `firwin`.
- **Phase Alignment**: Forward-backward zero-phase filtering (`phase="zero"`), ensuring zero phase shift across all channels and sub-bands.
- **Transition Bandwidth**: Automatically adjusted based on sampling rate ($250\text{ Hz}$) and window duration to prevent ringing artifacts.

### Tensor Shape Transformations

The transform handles both single time-domain windows and batches of trials/windows seamlessly:

```
Single Window Input : (Channels, Samples)         ---> (Bands, Channels, Samples)
                      (133, 250)                  ---> (4, 133, 250)

Batch Input         : (N, Channels, Samples)      ---> (N, Bands, Channels, Samples)
                      (3520, 133, 250)            ---> (3520, 4, 133, 250)
```

### Representation Modes

The pipeline and PyTorch dataset support two primary representation modes:

1. **Time Domain Mode (`representation="time"`)**:
   - Returns standard 3D window tensors `(N_windows, Channels, Samples)`.
   - Preserves 100% numerical and functional equivalence with the baseline pipeline (`basefile.py`).

2. **Frequency Domain Mode (`representation="frequency"`)**:
   - Returns 4D multi-band tensors `(N_windows, Bands, Channels, Samples)`.
   - Each sample in PyTorch `HGDDataset` yields a tensor of shape `[Bands, Channels, Samples]`.

### YAML Configuration (`configs/preprocessing.yaml`)

Configuration is fully declarative and parsed into `FrequencyRepresentationConfig` and `FrequencyBandConfig`:

```yaml
frequency:
  enabled: false          # Set representation="frequency" to enable
  fir_design: "firwin"     # Design method for mne.filter.filter_data
  bands:
    - name: theta
      low: 4.0
      high: 8.0
    - name: alpha
      low: 8.0
      high: 13.0
    - name: beta
      low: 13.0
      high: 30.0
    - name: gamma
      low: 30.0
      high: 38.0
```

### Code Usage Example

```python
from datasets import EEGPreprocessingPipeline, HGDDataset

# 1. Initialize Pipeline with configuration
pipeline = EEGPreprocessingPipeline(config="configs/preprocessing.yaml")

# 2. Extract multi-band frequency tensor: (N_windows, 4, 133, 250)
X_freq, y_windows, trial_ids = pipeline.process("hgd/train1/1.edf", representation="frequency")

# 3. Instantiate PyTorch Dataset wrapper
dataset = HGDDataset(
    file_paths="hgd/train1/1.edf",
    pipeline=pipeline,
    representation="frequency"
)

sample_tensor, label = dataset[0]
print("PyTorch sample tensor shape:", sample_tensor.shape)  # torch.Size([4, 133, 250])

# 4. Debug export mode (saves frequency_tensor.npy, frequency_metadata.json, frequency_summary.json)
debug_dict = pipeline.process_debug("hgd/train1/1.edf", representation="frequency")
```

### Verification & Testing

Run unit tests verifying configuration validation, tensor transformations, band ordering, absence of NaN/Inf values, and debug artifact export:

```bash
python -m unittest tests/test_frequency_representation.py
python -m unittest tests/test_pipeline_equivalence.py
python scripts/test_pipeline.py
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
git clone <repo-url>
cd <repo>

python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

python -m pip install -r requirements.txt

python scripts/profile_dataset.py
```

## License

This project is licensed under the [MIT License](LICENSE).
