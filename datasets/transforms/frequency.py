"""
Frequency-Aware EEG Representation Module.

Provides FrequencyRepresentation transform that decomposes EEG signals into
multi-band spectral representations (e.g., Theta, Alpha, Beta, Gamma) using
zero-phase FIR filtering via MNE (mne.filter.filter_data).

Tensor Shape Transformations:
    Single Window : (Channels, Samples)    -> (Bands, Channels, Samples)
    Batch Windows : (N, Channels, Samples) -> (N, Bands, Channels, Samples)
"""

import os
import json
import time
import logging
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
import mne

logger = logging.getLogger(__name__)


@dataclass
class FrequencyBandConfig:
    """Configuration for a single frequency band."""
    name: str
    low: float
    high: float


@dataclass
class FrequencyRepresentationConfig:
    """Configuration for multi-band frequency representation."""
    sampling_rate: float = 250.0
    bands: List[FrequencyBandConfig] = field(default_factory=lambda: [
        FrequencyBandConfig(name="theta", low=4.0, high=8.0),
        FrequencyBandConfig(name="alpha", low=8.0, high=13.0),
        FrequencyBandConfig(name="beta", low=13.0, high=30.0),
        FrequencyBandConfig(name="gamma", low=30.0, high=38.0),
    ])
    fir_design: str = "firwin"
    debug: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any], sampling_rate: float = 250.0) -> "FrequencyRepresentationConfig":
        """Construct FrequencyRepresentationConfig from a dictionary (e.g. loaded YAML)."""
        if not data:
            return cls(sampling_rate=sampling_rate)

        if "bands" in data:
            raw_bands = data.get("bands") or []
            band_objs = []
            for b in raw_bands:
                band_objs.append(FrequencyBandConfig(
                    name=str(b.get("name", "")),
                    low=float(b.get("low", 0.0)),
                    high=float(b.get("high", 0.0)),
                ))
            bands = band_objs
        else:
            bands = cls().bands

        return cls(
            sampling_rate=sampling_rate,
            bands=bands,
            fir_design=str(data.get("fir_design", "firwin")),
            debug=bool(data.get("debug", False)),
        )


@dataclass
class FrequencyMetadata:
    """Metadata describing generated multi-band frequency tensor."""
    sampling_rate: float
    frequency_bands: List[Dict[str, Any]]
    tensor_shape: List[int]
    dtype: str
    timestamp: str
    filter_type: str
    execution_time_seconds: float
    filter_parameters: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to JSON-serializable dictionary."""
        return asdict(self)


class FrequencyRepresentation:
    """
    Modular Frequency-Aware EEG Representation Transformer.

    Decomposes EEG windows or window batches into multi-band spectral tensors.
    """

    def __init__(self, config: Optional[FrequencyRepresentationConfig] = None):
        """
        Initialize FrequencyRepresentation with configuration and validate parameters.

        Args:
            config: Optional FrequencyRepresentationConfig object. Uses defaults if None.
        """
        self.config = config if config is not None else FrequencyRepresentationConfig()
        self._validate_config()
        logger.info(
            f"Initialized FrequencyRepresentation with {len(self.config.bands)} sub-bands: "
            f"{[b.name for b in self.config.bands]} (fs={self.config.sampling_rate} Hz)"
        )

    def _validate_config(self) -> None:
        """
        Validate frequency band configuration bounds.

        Raises:
            ValueError: If low >= high, low <= 0, or high >= Nyquist frequency.
        """
        nyquist = self.config.sampling_rate / 2.0
        if not self.config.bands:
            err_msg = "Frequency representation configuration contains 0 frequency bands."
            logger.error(f"Configuration validation failed: {err_msg}")
            raise ValueError(err_msg)

        for b in self.config.bands:
            if b.low <= 0:
                err_msg = f"Invalid low frequency for band '{b.name}': {b.low} Hz must be > 0."
                logger.error(f"Configuration validation failed: {err_msg}")
                raise ValueError(err_msg)
            if b.low >= b.high:
                err_msg = (
                    f"Invalid band bounds for '{b.name}': low cutoff ({b.low} Hz) "
                    f"must be strictly less than high cutoff ({b.high} Hz)."
                )
                logger.error(f"Configuration validation failed: {err_msg}")
                raise ValueError(err_msg)
            if b.high >= nyquist:
                err_msg = (
                    f"Invalid high cutoff for band '{b.name}': {b.high} Hz exceeds or equals "
                    f"Nyquist frequency ({nyquist} Hz) for sampling rate {self.config.sampling_rate} Hz."
                )
                logger.error(f"Configuration validation failed: {err_msg}")
                raise ValueError(err_msg)

        logger.info("Frequency representation configuration validation succeeded.")

    def filter_band(self, signal: np.ndarray, band: FrequencyBandConfig) -> np.ndarray:
        """
        Apply zero-phase FIR bandpass filter for a single frequency band.

        Args:
            signal (np.ndarray): Input EEG signal array of shape (..., Channels, Samples).
            band (FrequencyBandConfig): Frequency band specification.

        Returns:
            np.ndarray: Filtered EEG signal array of identical shape.
        """
        n_samples = signal.shape[-1]
        l_trans = max(2.0, band.low * 0.5)
        h_trans = max(2.0, band.high * 0.25)

        if n_samples < 400:
            l_trans = max(4.0, band.low * 0.5)
            h_trans = max(4.0, band.high * 0.25)

        filtered = mne.filter.filter_data(
            data=signal.astype(np.float64),
            sfreq=self.config.sampling_rate,
            l_freq=band.low,
            h_freq=band.high,
            l_trans_bandwidth=l_trans,
            h_trans_bandwidth=h_trans,
            filter_length="auto",
            method="fir",
            phase="zero",
            fir_design=self.config.fir_design,
            verbose=False
        )
        return filtered.astype(np.float32)

    def extract(self, window_or_batch: np.ndarray) -> Tuple[np.ndarray, FrequencyMetadata]:
        """
        Decompose time-domain window or batch into multi-band frequency tensor.

        Input Shapes:
            Single Window : (Channels, Samples) -> e.g., (133, 250)
            Batch Windows : (N, Channels, Samples) -> e.g., (3520, 133, 250)

        Output Shapes:
            Single Window : (Bands, Channels, Samples) -> e.g., (4, 133, 250)
            Batch Windows : (N, Bands, Channels, Samples) -> e.g., (3520, 4, 133, 250)

        Args:
            window_or_batch (np.ndarray): Time-domain signal array (2D or 3D).

        Returns:
            Tuple[np.ndarray, FrequencyMetadata]:
                - Multi-band frequency tensor
                - Generated FrequencyMetadata object
        """
        if not isinstance(window_or_batch, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray input, got {type(window_or_batch)}")

        ndim = window_or_batch.ndim
        if ndim not in (2, 3):
            raise ValueError(
                f"Invalid input signal dimensions: expected 2D (Channels, Samples) "
                f"or 3D (N, Channels, Samples), got {ndim}D with shape {window_or_batch.shape}"
            )

        t0 = time.time()
        logger.info(f"Extracting frequency representation for input shape {window_or_batch.shape}...")

        band_tensors = []
        for b in self.config.bands:
            t_band0 = time.time()
            filtered_band = self.filter_band(window_or_batch, b)
            band_elapsed = time.time() - t_band0
            logger.info(f"  Band '{b.name}' ({b.low}-{b.high} Hz) extracted in {band_elapsed:.3f}s")
            band_tensors.append(filtered_band)

        # Stack along appropriate axis to maintain deterministic band ordering
        if ndim == 2:
            # (Bands, Channels, Samples)
            out_tensor = np.stack(band_tensors, axis=0)
        else:
            # (N, Bands, Channels, Samples)
            out_tensor = np.stack(band_tensors, axis=1)

        total_elapsed = time.time() - t0

        # Validate Output Integrity
        if np.isnan(out_tensor).any():
            raise ValueError("NaN values detected in extracted frequency representation tensor.")
        if np.isinf(out_tensor).any():
            raise ValueError("Inf values detected in extracted frequency representation tensor.")

        # Construct Metadata
        metadata = FrequencyMetadata(
            sampling_rate=self.config.sampling_rate,
            frequency_bands=[asdict(b) for b in self.config.bands],
            tensor_shape=list(out_tensor.shape),
            dtype=str(out_tensor.dtype),
            timestamp=datetime.now().isoformat(),
            filter_type="FIR (zero-phase)",
            execution_time_seconds=round(total_elapsed, 4),
            filter_parameters={"fir_design": self.config.fir_design, "phase": "zero"},
        )

        logger.info(
            f"Frequency extraction completed in {total_elapsed:.3f}s. Output shape: {out_tensor.shape}"
        )

        return out_tensor, metadata

    def export_debug(self, freq_tensor: np.ndarray, metadata: FrequencyMetadata, debug_dir: str) -> None:
        """
        Export debug artifacts to disk.

        Saves:
        - frequency_tensor.npy
        - frequency_metadata.json
        - frequency_summary.json
        """
        os.makedirs(debug_dir, exist_ok=True)

        tensor_path = os.path.join(debug_dir, "frequency_tensor.npy")
        metadata_path = os.path.join(debug_dir, "frequency_metadata.json")
        summary_path = os.path.join(debug_dir, "frequency_summary.json")

        np.save(tensor_path, freq_tensor)

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata.to_dict(), f, indent=4)

        summary = {
            "bands": len(self.config.bands),
            "band_names": [b.name for b in self.config.bands],
            "shape": list(freq_tensor.shape),
            "dtype": str(freq_tensor.dtype),
            "sampling_rate": self.config.sampling_rate,
            "execution_time_seconds": metadata.execution_time_seconds,
        }

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

        logger.info(f"Exported frequency debug artifacts to {debug_dir}")
