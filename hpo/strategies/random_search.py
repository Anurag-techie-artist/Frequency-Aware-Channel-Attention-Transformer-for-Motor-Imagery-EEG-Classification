"""
Random Search Strategy Implementation.
"""

import random
from typing import Dict, Any
from hpo.search_space import SearchSpace
from hpo.strategies.base import SearchStrategy


class RandomSearchStrategy(SearchStrategy):
    """Samples hyperparameter combinations uniformly at random from search space."""

    def __init__(self, seed: int = 42):
        super().__init__(seed=seed)
        self.rng = random.Random(seed)

    def suggest(self, search_space: SearchSpace, trial_id: int) -> Dict[str, Any]:
        """Suggest random parameter combination for trial."""
        return search_space.sample(rng=self.rng)
