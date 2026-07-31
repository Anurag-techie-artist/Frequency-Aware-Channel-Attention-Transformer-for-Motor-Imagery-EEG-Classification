"""
HPOExperimentRunner Orchestrator Module.

Loads merged master configuration and launches HyperparameterOptimizer execution engine.
Implements BaseExperiment unified lifecycle contract.
"""

import os
import json
import logging
from typing import Dict, Any, Optional

from configs.config_loader import load_master_config
from experiments.base import BaseExperiment
from hpo.optimizer import HyperparameterOptimizer
from hpo.scheduler import TrialScheduler

logger = logging.getLogger(__name__)


class HPOExperimentRunner(BaseExperiment):
    """Orchestrates hyperparameter optimization experiment run adhering to BaseExperiment interface."""

    def __init__(self, hpo_config_path: Optional[str] = None):
        self.config = load_master_config(train_cfg_path=hpo_config_path)
        self.optimizer: Optional[HyperparameterOptimizer] = None
        self.scheduler: Optional[TrialScheduler] = None

    def run(self, resume: bool = True, **kwargs) -> TrialScheduler:
        """Execute HPO experiment run."""
        self.optimizer = HyperparameterOptimizer(master_config=self.config)
        logger.info("Launching Hyperparameter Optimizer Experiment...")
        self.scheduler = self.optimizer.optimize(resume=resume)
        return self.scheduler

    def resume(self, checkpoint_path_or_dir: str = None, **kwargs) -> TrialScheduler:
        """Resume HPO experiment run."""
        return self.run(resume=True, **kwargs)

    def summarize(self) -> Dict[str, Any]:
        """Summarize HPO experiment results."""
        out_dir = str(self.config.get("hpo", {}).get("output_dir", "outputs/hpo"))
        summary_path = os.path.join(out_dir, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                return json.load(f)

        if self.scheduler:
            best_t = self.scheduler.get_best_trial()
            return {
                "total_trials": len(self.scheduler.trials),
                "best_trial_id": best_t.trial_id if best_t else None,
                "best_score": best_t.score if best_t else None,
            }
        return {"status": "no_summary_available"}
