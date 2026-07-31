"""
HyperparameterOptimizer Module.

Executes trial runs, updates SearchStrategy, manages TrialScheduler queue, records ObjectiveResult,
exports trial manifests, and generates leaderboard summaries and visualization plots.
"""

import os
import time
import logging
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import pandas as pd

from hpo.search_space import SearchSpace
from hpo.factory import build_hpo_strategy
from hpo.scheduler import TrialScheduler
from hpo.trial import Trial, TrialStatus
from hpo.objective import ObjectiveResult
from hpo.results import HPOResultsManager
from hpo.visualization import plot_optimization_history, plot_parallel_coordinates

from models.eeg_motor_imagery_model import EEGMotorImageryModel
from datasets.builder import build_dataloaders
from training import build_loss, build_optimizer, build_scheduler, Trainer, get_device

logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """Orchestrates hyperparameter trial execution without modifying core model/trainer code."""

    def __init__(self, master_config: Dict[str, Any]):
        self.master_config = master_config
        self.hpo_config = master_config.get("hpo", {})
        self.search_space_config = master_config.get("search_space", {})

        self.search_space = SearchSpace(self.search_space_config)
        self.strategy = build_hpo_strategy(master_config)

        self.n_trials = int(self.hpo_config.get("n_trials", 20))
        self.metric_name = str(self.hpo_config.get("metric", "val_accuracy"))
        self.mode = str(self.hpo_config.get("mode", "max")).lower()
        self.max_epochs = int(self.hpo_config.get("max_epochs_per_trial", 5))
        self.output_dir = str(self.hpo_config.get("output_dir", "outputs/hpo"))

        self.scheduler = TrialScheduler(max_trials=self.n_trials)
        self.results_manager = HPOResultsManager(output_dir=self.output_dir)

    def _merge_trial_config(self, trial_params: Dict[str, Any]) -> Dict[str, Any]:
        """Merge trial parameters into master configuration dictionary."""
        merged = dict(self.master_config)
        train_cfg = dict(merged.get("training", {}))
        model_cfg = dict(merged.get("model", {}))

        # Override max epochs per trial
        train_cfg["epochs"] = self.max_epochs

        for p_name, p_val in trial_params.items():
            if p_name in train_cfg:
                train_cfg[p_name] = p_val
            elif p_name in model_cfg:
                model_cfg[p_name] = p_val
            else:
                # Search nested sub-configs
                for sub_key in ["attention", "transformer", "classifier"]:
                    if sub_key in model_cfg and isinstance(model_cfg[sub_key], dict):
                        if p_name in model_cfg[sub_key]:
                            sub_dict = dict(model_cfg[sub_key])
                            sub_dict[p_name] = p_val
                            model_cfg[sub_key] = sub_dict

        merged["training"] = train_cfg
        merged["model"] = model_cfg
        return merged

    def run_trial_objective(self, trial_id: int, trial_params: Dict[str, Any]) -> ObjectiveResult:
        """Execute a single trial training run and compute objective result."""
        start_time = time.perf_counter()
        trial_config = self._merge_trial_config(trial_params)

        train_cfg = trial_config.get("training", {})
        device = get_device(train_cfg.get("device", "auto"))

        model = EEGMotorImageryModel.from_config(trial_config)
        criterion = build_loss(trial_config)
        optimizer = build_optimizer(model, trial_config)
        scheduler = build_scheduler(optimizer, trial_config)

        train_loader, val_loader, _ = build_dataloaders(trial_config)

        # Trial checkpoint output path
        trial_ckpt_dir = os.path.join(self.output_dir, "trials", f"trial_{trial_id:03d}")
        os.makedirs(trial_ckpt_dir, exist_ok=True)
        trial_config["checkpoint"] = {"save_dir": trial_ckpt_dir, "save_best": True, "monitor": self.metric_name}

        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            config=trial_config,
            device=device,
        )

        state = trainer.fit(train_loader, val_loader)
        duration = time.perf_counter() - start_time

        val_metrics = trainer.validate_epoch(val_loader)
        score = float(val_metrics.get(self.metric_name.replace("val_", ""), val_metrics.get("accuracy", 0.0)))
        if self.metric_name == "val_loss":
            score = float(val_metrics.get("loss", score))

        best_ckpt = os.path.join(trial_ckpt_dir, "best.pt")
        return ObjectiveResult(
            score=score,
            metrics=val_metrics,
            checkpoint_path=best_ckpt if os.path.exists(best_ckpt) else None,
            duration_seconds=duration,
        )

    def optimize(self, resume: bool = True) -> TrialScheduler:
        """
        Execute hyperparameter optimization over specified number of trials.

        Args:
            resume: If True, resumes optimization from previously saved trials

        Returns:
            Populated TrialScheduler object
        """
        logger.info(f"Starting HPO run ({self.n_trials} trials, strategy: {self.hpo_config.get('strategy', 'random')})...")

        # Resume logic
        start_id = 0
        trials_csv = os.path.join(self.output_dir, "trials.csv")
        if resume and os.path.exists(trials_csv) and os.path.getsize(trials_csv) > 0:
            try:
                df_prev = pd.read_csv(trials_csv)
                if not df_prev.empty and "trial_id" in df_prev.columns:
                    start_id = int(df_prev["trial_id"].max()) + 1
                    logger.info(f"Resumed HPO run from trial_id {start_id}")
                    # Re-populate existing trials in scheduler
                    for _, row in df_prev.iterrows():
                        tid = int(row["trial_id"])
                        score_val = float(row["score"]) if "score" in row and not pd.isna(row["score"]) else None
                        params = {col.replace("param_", ""): row[col] for col in df_prev.columns if col.startswith("param_")}
                        status_str = str(row.get("status", "COMPLETED")).upper()
                        status = getattr(TrialStatus, status_str, TrialStatus.COMPLETED)

                        prev_trial = Trial(
                            trial_id=tid,
                            params=params,
                            status=status,
                            score=score_val,
                            duration_seconds=float(row.get("duration_seconds", 0.0)),
                        )
                        self.scheduler.trials.append(prev_trial)
            except Exception as e:
                logger.warning(f"Could not parse previous HPO trials CSV for resume: {e}")

        for t_id in range(start_id, self.n_trials):
            params = self.strategy.suggest(self.search_space, trial_id=t_id)
            trial = self.scheduler.add_trial(trial_id=t_id, params=params)

            self.scheduler.mark_running(t_id)
            logger.info(f"Executing HPO Trial {t_id:03d}/{self.n_trials:03d} with params: {params}")

            try:
                obj_res = self.run_trial_objective(trial_id=t_id, trial_params=params)
                completed_trial = self.scheduler.mark_completed(
                    trial_id=t_id,
                    score=obj_res.score,
                    metrics=obj_res.metrics,
                    checkpoint_path=obj_res.checkpoint_path,
                    duration_seconds=obj_res.duration_seconds,
                )
                self.strategy.update(completed_trial)
                self.results_manager.export_trial_manifest(completed_trial, self.master_config)
                logger.info(f"Trial {t_id:03d} COMPLETED | Score ({self.metric_name}): {obj_res.score:.4f}")
            except Exception as e:
                logger.error(f"Trial {t_id:03d} FAILED with error: {e}")
                failed_trial = self.scheduler.mark_failed(trial_id=t_id, error_message=str(e))
                self.strategy.update(failed_trial)
                self.results_manager.export_trial_manifest(failed_trial, self.master_config)

        # Export leaderboard, best config, and summary
        best_trial = self.scheduler.get_best_trial(mode=self.mode)
        out_paths = self.results_manager.export_summary_and_leaderboard(
            trials=self.scheduler.trials,
            best_trial=best_trial,
            master_config=self.master_config,
            mode=self.mode,
        )

        # Generate plots if trials exist
        if os.path.exists(out_paths["trials_csv"]) and os.path.getsize(out_paths["trials_csv"]) > 0:
            plot_optimization_history(
                out_paths["trials_csv"],
                metric_name=self.metric_name,
                save_path=os.path.join(self.output_dir, "optimization_history.png"),
            )
            plot_parallel_coordinates(
                out_paths["trials_csv"],
                save_path=os.path.join(self.output_dir, "parallel_coordinates.png"),
            )

        if best_trial:
            logger.info(f"HPO Run Finished! Best Trial: #{best_trial.trial_id} | Score: {best_trial.score:.4f}")
        return self.scheduler
