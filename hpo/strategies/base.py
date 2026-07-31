"""
Abstract Base Class for HPO Search Strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from hpo.search_space import SearchSpace
from hpo.trial import Trial


class SearchStrategy(ABC):
    """Abstract Base Class for hyperparameter search strategies."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    @abstractmethod
    def suggest(self, search_space: SearchSpace, trial_id: int) -> Dict[str, Any]:
        """Suggest parameters for the next trial."""
        pass

    def update(self, trial: Trial):
        """Update internal strategy state with completed trial results."""
        pass
