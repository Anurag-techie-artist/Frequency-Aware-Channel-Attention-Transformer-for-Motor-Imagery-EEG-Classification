"""
Abstract Base Class for HPO Search Space Parameters.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class Parameter(ABC):
    """Abstract Base Class for search space parameter types."""

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config

    @abstractmethod
    def sample(self, rng: Any = None) -> Any:
        """Sample a value for this parameter."""
        pass

    @abstractmethod
    def validate(self) -> bool:
        """Validate parameter definition settings."""
        pass
