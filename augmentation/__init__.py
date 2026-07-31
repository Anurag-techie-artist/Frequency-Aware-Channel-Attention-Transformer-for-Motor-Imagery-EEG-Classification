"""
Research Augmentation Package for Motor Imagery EEG Classification.
"""

from augmentation.registry import AUGMENTATION_REGISTRY, register_augmentation_strategy
from augmentation.factory import build_augmentation_strategy
from augmentation.pipeline import AugmentationPipeline
from augmentation.dataset import SyntheticDataset, SimpleSyntheticDataset, AugmentedTensorDataset
from augmentation.validator import SyntheticDataValidator
from augmentation.artifacts import ArtifactManager
from augmentation.runner import AugmentationExperimentRunner
from augmentation.ablation import AugmentationRatioAblationRunner

__all__ = [
    "AUGMENTATION_REGISTRY",
    "register_augmentation_strategy",
    "build_augmentation_strategy",
    "AugmentationPipeline",
    "SyntheticDataset",
    "SimpleSyntheticDataset",
    "AugmentedTensorDataset",
    "SyntheticDataValidator",
    "ArtifactManager",
    "AugmentationExperimentRunner",
    "AugmentationRatioAblationRunner",
]
