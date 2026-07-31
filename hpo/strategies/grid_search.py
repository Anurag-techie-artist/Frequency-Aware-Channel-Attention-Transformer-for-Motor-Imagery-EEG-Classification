"""
Grid Search Strategy Implementation.
"""

import itertools
from typing import Dict, Any, List
from hpo.search_space import SearchSpace
from hpo.strategies.base import SearchStrategy
from hpo.parameters.categorical import CategoricalParameter


class GridSearchStrategy(SearchStrategy):
    """Enumerates Cartesian product grid of parameter options."""

    def __init__(self, seed: int = 42):
        super().__init__(seed=seed)
        self.grid: List[Dict[str, Any]] = []
        self._grid_initialized = False

    def _build_grid(self, search_space: SearchSpace):
        param_options = []
        param_names = []

        for name, param in search_space.parameters.items():
            param_names.append(name)
            if isinstance(param, CategoricalParameter):
                options = param.values
            else:
                # Sample 3 discrete grid points for continuous params
                options = [
                    param.low,
                    (param.low + param.high) / 2.0,
                    param.high,
                ]
            param_options.append(options)

        combo_tuples = list(itertools.product(*param_options))
        self.grid = [
            dict(zip(param_names, combo)) for combo in combo_tuples
        ]
        self._grid_initialized = True

    def suggest(self, search_space: SearchSpace, trial_id: int) -> Dict[str, Any]:
        """Suggest next grid point combination."""
        if not self._grid_initialized:
            self._build_grid(search_space)

        if trial_id < len(self.grid):
            return self.grid[trial_id]

        # Wrap around if trial_id exceeds grid size
        idx = trial_id % len(self.grid)
        return self.grid[idx]
