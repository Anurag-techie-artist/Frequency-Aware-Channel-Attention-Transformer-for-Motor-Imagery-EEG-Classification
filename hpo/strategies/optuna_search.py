"""
Optuna Search Strategy Wrapper with Graceful Fallback.
"""

import logging
from typing import Dict, Any, Optional
from hpo.search_space import SearchSpace
from hpo.strategies.base import SearchStrategy
from hpo.strategies.random_search import RandomSearchStrategy
from hpo.trial import Trial

logger = logging.getLogger(__name__)


class OptunaSearchStrategy(SearchStrategy):
    """Wraps Optuna TPE sampler if installed, with RandomSearch fallback."""

    def __init__(self, seed: int = 42):
        super().__init__(seed=seed)
        self.optuna_available = False
        self.study = None
        self.fallback = RandomSearchStrategy(seed=seed)

        try:
            import optuna

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            sampler = optuna.samplers.TPESampler(seed=seed)
            self.study = optuna.create_study(direction="maximize", sampler=sampler)
            self.optuna_available = True
        except ImportError:
            logger.warning(
                "Optuna package is not installed. Falling back to RandomSearchStrategy."
            )

    def suggest(self, search_space: SearchSpace, trial_id: int) -> Dict[str, Any]:
        """Suggest parameter combination using Optuna TPE or RandomSearch fallback."""
        if not self.optuna_available or self.study is None:
            return self.fallback.suggest(search_space, trial_id)

        import optuna

        optuna_trial = self.study.ask()
        params = {}

        for name, param in search_space.parameters.items():
            ptype = param.config.get("type", "float").lower()
            if ptype == "categorical":
                params[name] = optuna_trial.suggest_categorical(name, param.values)
            elif ptype == "integer" or ptype == "int":
                params[name] = optuna_trial.suggest_int(
                    name, param.low, param.high, step=getattr(param, "step", 1)
                )
            elif ptype == "loguniform":
                params[name] = optuna_trial.suggest_float(
                    name, param.low, param.high, log=True
                )
            else:
                is_log = param.config.get("distribution", "uniform").lower() == "loguniform"
                params[name] = optuna_trial.suggest_float(
                    name, param.low, param.high, log=is_log
                )

        return params

    def update(self, trial: Trial):
        """Update Optuna study with completed trial score."""
        if self.optuna_available and self.study is not None and trial.score is not None:
            try:
                # Tell optuna trial result
                self.study.tell(trial.trial_id, trial.score)
            except Exception as e:
                logger.debug(f"Optuna update note: {e}")
