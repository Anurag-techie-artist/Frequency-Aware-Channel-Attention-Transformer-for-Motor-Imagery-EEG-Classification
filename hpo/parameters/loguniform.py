"""
LogUniform Parameter Alias Search Space Definition.
"""

from typing import Any, Dict
from hpo.parameters.float import FloatParameter


class LogUniformParameter(FloatParameter):
    """Alias parameter type explicitly configuring a loguniform float parameter."""

    def __init__(self, name: str, config: Dict[str, Any]):
        cfg = dict(config)
        cfg["distribution"] = "loguniform"
        super().__init__(name, cfg)
