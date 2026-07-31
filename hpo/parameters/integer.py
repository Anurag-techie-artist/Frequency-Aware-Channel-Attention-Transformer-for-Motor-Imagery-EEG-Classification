"""
Integer Parameter Search Space Definition.
"""

import random
from typing import Any, Dict
from hpo.parameters.base import Parameter


class IntegerParameter(Parameter):
    """Integer parameter type."""

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.low = int(config.get("low", 1))
        self.high = int(config.get("high", 10))
        self.step = int(config.get("step", 1))
        self.validate()

    def validate(self) -> bool:
        if self.low > self.high:
            raise ValueError(
                f"Integer parameter '{self.name}' low bound ({self.low}) must be <= high bound ({self.high})."
            )
        if self.step <= 0:
            raise ValueError(
                f"Integer parameter '{self.name}' step size must be > 0."
            )
        return True

    def sample(self, rng: Any = None) -> int:
        r = rng if rng is not None else random
        options = list(range(self.low, self.high + 1, self.step))
        return int(r.choice(options))
