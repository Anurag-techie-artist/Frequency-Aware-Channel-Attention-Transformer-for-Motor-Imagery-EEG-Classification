"""
Hyperparameter Optimization (HPO) Package.

Extensible plugin package adhering to the Open/Closed Principle.
"""

from hpo.registry import PARAMETER_REGISTRY, STRATEGY_REGISTRY, register_parameter, register_strategy
from hpo.search_space import SearchSpace
from hpo.validator import SearchSpaceValidator
from hpo.trial import Trial, TrialStatus
from hpo.objective import ObjectiveResult
from hpo.scheduler import TrialScheduler
from hpo.factory import build_hpo_strategy
from hpo.optimizer import HyperparameterOptimizer
from hpo.results import HPOResultsManager
from hpo.runner import HPOExperimentRunner

__all__ = [
    "PARAMETER_REGISTRY",
    "STRATEGY_REGISTRY",
    "register_parameter",
    "register_strategy",
    "SearchSpace",
    "SearchSpaceValidator",
    "Trial",
    "TrialStatus",
    "ObjectiveResult",
    "TrialScheduler",
    "build_hpo_strategy",
    "HyperparameterOptimizer",
    "HPOResultsManager",
    "HPOExperimentRunner",
]
