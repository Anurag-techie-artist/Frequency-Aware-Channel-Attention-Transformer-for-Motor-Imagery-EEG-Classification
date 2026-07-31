"""
Global Registries for HPO Parameter Types and Search Strategies.

Enables Open/Closed Principle: adding new parameter types or search strategies requires
registering them in these dictionaries without modifying core optimization logic.
"""

from typing import Dict, Type, Any

from hpo.parameters.base import Parameter
from hpo.parameters.float import FloatParameter
from hpo.parameters.integer import IntegerParameter
from hpo.parameters.categorical import CategoricalParameter
from hpo.parameters.loguniform import LogUniformParameter

PARAMETER_REGISTRY: Dict[str, Type[Parameter]] = {
    "float": FloatParameter,
    "integer": IntegerParameter,
    "int": IntegerParameter,
    "categorical": CategoricalParameter,
    "loguniform": LogUniformParameter,
}

STRATEGY_REGISTRY: Dict[str, Type[Any]] = {}


def register_parameter(name: str, parameter_cls: Type[Parameter]):
    """Register a new custom parameter type."""
    PARAMETER_REGISTRY[name.lower()] = parameter_cls


def register_strategy(name: str, strategy_cls: Type[Any]):
    """Register a new custom search strategy."""
    STRATEGY_REGISTRY[name.lower()] = strategy_cls
