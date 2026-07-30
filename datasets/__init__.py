"""
HGD Preprocessing Package Exports.
"""

from datasets.loader import (
    locate_dataset,
    discover_edf_files,
    load_raw_edf,
    extract_annotations,
    extract_events,
    validate_raw,
)
from datasets.preprocessing import (
    resample_signal,
    bandpass_filter,
    select_channels,
    normalize_signal,
    extract_epochs,
    preprocess_recording,
)
from datasets.windowing import (
    calculate_window_indices,
    window_labels,
    generate_sliding_windows,
)
from datasets.pipeline import (
    EEGPreprocessingPipeline,
    PreprocessingConfig,
)
from datasets.dataset import HGDDataset
from datasets.transforms import (
    FrequencyBandConfig,
    FrequencyRepresentationConfig,
    FrequencyMetadata,
    FrequencyRepresentation,
)

__all__ = [
    "locate_dataset",
    "discover_edf_files",
    "load_raw_edf",
    "extract_annotations",
    "extract_events",
    "validate_raw",
    "resample_signal",
    "bandpass_filter",
    "select_channels",
    "normalize_signal",
    "extract_epochs",
    "preprocess_recording",
    "calculate_window_indices",
    "window_labels",
    "generate_sliding_windows",
    "EEGPreprocessingPipeline",
    "PreprocessingConfig",
    "HGDDataset",
    "FrequencyBandConfig",
    "FrequencyRepresentationConfig",
    "FrequencyMetadata",
    "FrequencyRepresentation",
]
