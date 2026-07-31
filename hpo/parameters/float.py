"""
Float Parameter Search Space Definition.
"""

import random
from typing import Any, Dict
from hpo.parameters.base import Parameter


class FloatParameter(Parameter):
    """Float parameter type (supports uniform and loguniform distributions)."""

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.low = float(config.get("low", 0.0))
        self.high = float(config.get("high", 1.0))
        self.distribution = config.get("distribution", "uniform").lower()
        self.validate()

    def validate(self) -> bool:
        if self.low >= self.high:
            raise ValueError(
                f"Float parameter '{self.name}' low bound ({self.low}) must be strictly less than high bound ({self.high})."
            )
        if self.distribution == "loguniform" and self.low <= 0:
            raise ValueError(
                f"Float parameter '{self.name}' with loguniform distribution must have low bound > 0."
            )
        return True

    def sample(self, rng: Any = None) -> float:
        r = rng if rng is not None else random
        if self.distribution == "loguniform":
            import math

            log_low = math.log(self.low)
            log_high = math.log(self.high)
            return float(math.exp(r.uniform(log_low, log_high)))
        return float(r.uniform(self.low, self.high))
