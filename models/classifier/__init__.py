"""
Classifier heads and motor imagery prediction modules.
"""

from models.classifier.classification_head import (
    ClassificationHead,
    ClassificationHeadConfig,
)
from models.classifier.eeg_classifier import (
    EEGClassifier,
    ClassifierConfig,
    ClassifierMetadata,
    PredictionOutput,
)

__all__ = [
    "ClassificationHead",
    "ClassificationHeadConfig",
    "EEGClassifier",
    "ClassifierConfig",
    "ClassifierMetadata",
    "PredictionOutput",
]
