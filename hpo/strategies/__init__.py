"""
Strategies Package & Registry Initialization.
"""

from hpo.registry import register_strategy, STRATEGY_REGISTRY
from hpo.strategies.base import SearchStrategy
from hpo.strategies.random_search import RandomSearchStrategy
from hpo.strategies.grid_search import GridSearchStrategy
from hpo.strategies.optuna_search import OptunaSearchStrategy

# Register strategy implementations
register_strategy("random", RandomSearchStrategy)
register_strategy("grid", GridSearchStrategy)
register_strategy("optuna", OptunaSearchStrategy)

__all__ = [
    "SearchStrategy",
    "RandomSearchStrategy",
    "GridSearchStrategy",
    "OptunaSearchStrategy",
]
