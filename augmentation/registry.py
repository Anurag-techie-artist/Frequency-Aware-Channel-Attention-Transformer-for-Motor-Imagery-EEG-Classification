"""
AUGMENTATION_REGISTRY for Open/Closed Strategy Registration.

Enables adding new augmentation methods (MixUp, CutMix, SMOTE, WGAN-GP, Diffusion)
without modifying existing training or evaluation infrastructure.
"""

from typing import Dict, Type, Any
from augmentation.strategies.base import AugmentationStrategy
from augmentation.strategies.none import NoAugmentationStrategy
from augmentation.strategies.wgan_gp import WGANGPStrategy
from augmentation.strategies.mixup import MixUpStrategy
from augmentation.strategies.cutmix import CutMixStrategy
from augmentation.strategies.smote import SMOTEStrategy
from augmentation.strategies.diffusion import DiffusionStrategy

AUGMENTATION_REGISTRY: Dict[str, Type[AugmentationStrategy]] = {
    "none": NoAugmentationStrategy,
    "wgan_gp": WGANGPStrategy,
    "mixup": MixUpStrategy,
    "cutmix": CutMixStrategy,
    "smote": SMOTEStrategy,
    "diffusion": DiffusionStrategy,
}


def register_augmentation_strategy(name: str, strategy_cls: Type[AugmentationStrategy]):
    """Register a new custom augmentation strategy."""
    AUGMENTATION_REGISTRY[name.lower()] = strategy_cls
