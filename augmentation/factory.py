"""
Augmentation Strategy Factory Module.
"""

from typing import Dict, Any
from augmentation.registry import AUGMENTATION_REGISTRY
from augmentation.strategies.base import AugmentationStrategy


def build_augmentation_strategy(config: Dict[str, Any]) -> AugmentationStrategy:
    """
    Build augmentation strategy instance from configuration dictionary using AUGMENTATION_REGISTRY.

    Args:
        config: Master or augmentation configuration dictionary

    Returns:
        Instantiated AugmentationStrategy object
    """
    aug_cfg = config.get("augmentation", config)
    strategy_name = str(aug_cfg.get("strategy", "wgan_gp")).lower()
    seed = int(aug_cfg.get("seed", 42))

    if strategy_name not in AUGMENTATION_REGISTRY:
        raise ValueError(
            f"Unsupported strategy '{strategy_name}'. Supported strategies: {list(AUGMENTATION_REGISTRY.keys())}"
        )

    strategy_cls = AUGMENTATION_REGISTRY[strategy_name]
    return strategy_cls(seed=seed)
