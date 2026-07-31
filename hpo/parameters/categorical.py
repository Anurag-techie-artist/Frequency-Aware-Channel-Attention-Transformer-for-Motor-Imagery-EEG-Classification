"""
Categorical Parameter Search Space Definition.
"""

import random
from typing import Any, Dict, List
from hpo.parameters.base import Parameter


class CategoricalParameter(Parameter):
    """Categorical parameter type."""

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.values = list(config.get("values", []))
        self.validate()

    def validate(self) -> bool:
        if not self.values:
            raise ValueError(
                f"Categorical parameter '{self.name}' must provide a non-empty 'values' list."
            )
        return True

    def sample(self, rng: Any = None) -> Any:
        r = rng if rng is not None else random
        val = r.choice(self.values)
        return val
