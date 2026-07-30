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
