"""
HPO Strategy Factory Module.
"""

from typing import Dict, Any
from hpo.registry import STRATEGY_REGISTRY
from hpo.strategies import RandomSearchStrategy, SearchStrategy


def build_hpo_strategy(config: Dict[str, Any]) -> SearchStrategy:
    """
    Build search strategy instance from configuration dictionary using STRATEGY_REGISTRY.

    Args:
        config: Master or HPO configuration dictionary

    Returns:
        Instantiated SearchStrategy object
    """
    hpo_cfg = config.get("hpo", config)
    strategy_name = str(hpo_cfg.get("strategy", "random")).lower()
    seed = int(hpo_cfg.get("seed", 42))

    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unsupported strategy '{strategy_name}'. Supported strategies: {list(STRATEGY_REGISTRY.keys())}"
        )

    strategy_cls = STRATEGY_REGISTRY[strategy_name]
    return strategy_cls(seed=seed)
