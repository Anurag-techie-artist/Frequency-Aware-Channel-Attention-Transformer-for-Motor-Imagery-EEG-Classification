"""
EEG Preprocessing Pipeline Orchestrator Class.

Provides the extensible EEGPreprocessingPipeline base class that encapsulates
the full end-to-end data pipeline:
    load -> resample -> filter -> epoch -> normalize -> window -> representation
"""

import os
import yaml
import logging
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, Union, List

import numpy as np
import mne

from datasets.loader import load_raw_edf, extract_events
from datasets.preprocessing import resample_signal, bandpass_filter, extract_epochs
from datasets.windowing import generate_sliding_windows
from datasets.transforms.frequency import FrequencyRepresentation, FrequencyRepresentationConfig

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Dataclass holding preprocessing parameters."""
    sampling_rate: float = 250.0
    filter_low: float = 4.0
    filter_high: float = 38.0
    epoch_start: float = 0.5
    epoch_end: float = 3.5
    window_size: int = 250
    window_stride: int = 50
    normalization: str = "zscore"
    eps: float = 1e-6
    raw_dict: Optional[Dict[str, Any]] = None

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PreprocessingConfig":
        """Load configuration from a YAML file."""
        if not os.path.exists(yaml_path):
            logger.warning(f"Config file {yaml_path} not found. Using default baseline values.")
            return cls()

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls(
            sampling_rate=float(data.get("sampling_rate", 250.0)),
            filter_low=float(data.get("filter_low", 4.0)),
            filter_high=float(data.get("filter_high", 38.0)),
            epoch_start=float(data.get("epoch_start", 0.5)),
            epoch_end=float(data.get("epoch_end", 3.5)),
            window_size=int(data.get("window_size", 250)),
            window_stride=int(data.get("window_stride", 50)),
            normalization=str(data.get("normalization", "zscore")),
            eps=float(data.get("eps", 1e-6)),
            raw_dict=data,
        )


class EEGPreprocessingPipeline:
    """
    Extensible EEG Preprocessing Pipeline Class.

    Encapsulates sequential stages:
    1. load_raw
    2. resample
    3. filter
    4. epoch
    5. frequency representation (optional)
    6. window
    """

    def __init__(self, config: Optional[Union[PreprocessingConfig, str, Dict[str, Any]]] = None):
        """
        Initialize pipeline with configuration.

        Args:
            config: Can be a PreprocessingConfig object, file path to YAML, config dict, or None.
        """
        if config is None:
            default_yaml = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "configs", "preprocessing.yaml")
            )
            self.config = PreprocessingConfig.from_yaml(default_yaml)
        elif isinstance(config, str):
            self.config = PreprocessingConfig.from_yaml(config)
        elif isinstance(config, dict):
            self.config = PreprocessingConfig(**config)
        elif isinstance(config, PreprocessingConfig):
            self.config = config
        else:
            raise TypeError(f"Invalid config type: {type(config)}")

        # Initialize FrequencyRepresentation module
        freq_raw_dict = {}
        if self.config.raw_dict and "frequency" in self.config.raw_dict:
            freq_raw_dict = self.config.raw_dict["frequency"]

        freq_config = FrequencyRepresentationConfig.from_dict(
            freq_raw_dict,
            sampling_rate=self.config.sampling_rate
        )
        self.frequency_representation = FrequencyRepresentation(config=freq_config)

        logger.info(f"Initialized EEGPreprocessingPipeline with config: {self.config}")

    def load_raw(self, filepath: str) -> mne.io.Raw:
        """Stage 1: Load raw EDF file."""
        return load_raw_edf(filepath, preload=True)

    def resample(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Stage 2: Resample signal."""
        return resample_signal(raw, target_fs=self.config.sampling_rate)

    def filter(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Stage 3: Bandpass filter signal."""
        return bandpass_filter(
            raw,
            lowcut=self.config.filter_low,
            highcut=self.config.filter_high
        )

    def epoch(self, raw: mne.io.Raw) -> Tuple[np.ndarray, np.ndarray]:
        """Stage 4: Extract events and epochs."""
        events, event_dict = extract_events(raw)
        X_epochs, y_epochs = extract_epochs(
            raw,
            events,
            event_dict,
            tmin=self.config.epoch_start,
            tmax=self.config.epoch_end
        )
        return X_epochs, y_epochs

    def window(self, X_epochs: np.ndarray, y_epochs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Stage 5: Generate sliding windows and normalize."""
        normalize = (self.config.normalization == "zscore")
        return generate_sliding_windows(
            X_epochs,
            y_epochs,
            window_size=self.config.window_size,
            stride=self.config.window_stride,
            normalize=normalize,
            eps=self.config.eps
        )

    def process(self, filepath: str, representation: str = "time") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run end-to-end pipeline on an EDF file.

        Args:
            filepath (str): Path to EDF recording file.
            representation (str): "time" for (N, Channels, Samples) or "frequency" for (N, Bands, Channels, Samples).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - X_out: Windowed representation array
                - y_windows: (N_windows,)
                - trial_ids: (N_windows,)
        """
        if representation not in ("time", "frequency"):
            raise ValueError(f"Unknown representation mode '{representation}'. Supported modes: 'time', 'frequency'")

        raw = self.load_raw(filepath)
        raw = self.resample(raw)
        raw = self.filter(raw)
        X_epochs, y_epochs = self.epoch(raw)

        if representation == "frequency":
            logger.info("Applying FrequencyRepresentation transformation to epochs...")
            X_epochs_freq, _ = self.frequency_representation.extract(X_epochs)
            X_windows, y_windows, trial_ids = self.window(X_epochs_freq, y_epochs)
        else:
            X_windows, y_windows, trial_ids = self.window(X_epochs, y_epochs)

        return X_windows, y_windows, trial_ids

    def process_debug(self, filepath: str, representation: str = "time") -> Dict[str, Any]:
        """
        Run pipeline and return all intermediate stage outputs for inspection/debugging.

        Args:
            filepath (str): Path to EDF recording file.
            representation (str): "time" or "frequency".

        Returns:
            Dict[str, Any] containing stage outputs.
        """
        if representation not in ("time", "frequency"):
            raise ValueError(f"Unknown representation mode '{representation}'. Supported modes: 'time', 'frequency'")

        raw_obj = self.load_raw(filepath)
        raw_signal = raw_obj.get_data().copy()

        resampled_obj = self.resample(raw_obj)
        filtered_obj = self.filter(resampled_obj)
        filtered_signal = filtered_obj.get_data().copy()

        X_epochs, y_epochs = self.epoch(filtered_obj)

        if representation == "frequency":
            logger.info("Extracting frequency representation for debug export...")
            X_epochs_freq, freq_meta = self.frequency_representation.extract(X_epochs)
            X_windows, y_windows, trial_ids = self.window(X_epochs_freq, y_epochs)

            debug_res = {
                "raw": raw_signal,
                "filtered": filtered_signal,
                "epochs": X_epochs,
                "epochs_frequency": X_epochs_freq,
                "windows": X_windows,
                "labels": y_windows,
                "trial_ids": trial_ids,
                "frequency_tensor": X_windows,
                "frequency_metadata": freq_meta,
            }

            debug_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "outputs", "debug")
            )
            self.frequency_representation.export_debug(X_windows, freq_meta, debug_dir)
        else:
            X_windows, y_windows, trial_ids = self.window(X_epochs, y_epochs)
            debug_res = {
                "raw": raw_signal,
                "filtered": filtered_signal,
                "epochs": X_epochs,
                "windows": X_windows,
                "labels": y_windows,
                "trial_ids": trial_ids,
            }

        return debug_res
