"""
HPOResultsManager Module for Exporting Leaderboards, Summaries, and Trial Manifests.
"""

import os
import json
import yaml
import time
import pandas as pd
from typing import Dict, Any, List, Optional
from hpo.trial import Trial
from evaluation.manifest import get_git_commit_hash, compute_config_hash


class HPOResultsManager:
    """Exports trial manifests, leaderboard tables, best configuration YAMLs, and summaries."""

    def __init__(self, output_dir: str = "outputs/hpo"):
        self.output_dir = output_dir
        self.trials_dir = os.path.join(output_dir, "trials")
        os.makedirs(self.trials_dir, exist_ok=True)

    def export_trial_manifest(self, trial: Trial, config: Dict[str, Any]) -> str:
        """Export individual trial manifest.json."""
        trial_folder = os.path.join(self.trials_dir, f"trial_{trial.trial_id:03d}")
        os.makedirs(trial_folder, exist_ok=True)

        manifest = {
            "trial_id": trial.trial_id,
            "status": trial.status.name.lower(),
            "parameters": trial.params,
            "score": trial.score,
            "metrics": trial.metrics,
            "duration_seconds": trial.duration_seconds,
            "checkpoint": trial.checkpoint_path,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "git_commit": get_git_commit_hash(),
            "config_hash": compute_config_hash(config),
        }

        manifest_path = os.path.join(trial_folder, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        return manifest_path

    def export_summary_and_leaderboard(
        self,
        trials: List[Trial],
        best_trial: Optional[Trial],
        master_config: Dict[str, Any],
        mode: str = "max",
    ) -> Dict[str, str]:
        """Export leaderboard.csv, trials.csv, best_config.yaml, and summary.json."""
        rows = []
        for t in trials:
            row = {
                "trial_id": t.trial_id,
                "status": t.status.name,
                "score": t.score if t.score is not None else float("nan"),
                "duration_seconds": round(t.duration_seconds, 2),
            }
            # Flatten parameter values into table columns
            for p_name, p_val in t.params.items():
                row[f"param_{p_name}"] = p_val
            rows.append(row)

        df_all = pd.DataFrame(rows)
        trials_csv_path = os.path.join(self.output_dir, "trials.csv")
        df_all.to_csv(trials_csv_path, index=False)

        # Leaderboard sorted by score
        if not df_all.empty and "score" in df_all.columns:
            ascending = (mode.lower() == "min")
            df_leaderboard = df_all.sort_values(by="score", ascending=ascending)
        else:
            df_leaderboard = df_all

        leaderboard_csv_path = os.path.join(self.output_dir, "leaderboard.csv")
        df_leaderboard.to_csv(leaderboard_csv_path, index=False)

        # Export best configuration YAML
        best_config_path = os.path.join(self.output_dir, "best_config.yaml")
        if best_trial:
            merged_best_cfg = dict(master_config)
            model_cfg = dict(merged_best_cfg.get("model", {}))
            train_cfg = dict(merged_best_cfg.get("training", {}))

            for p_name, p_val in best_trial.params.items():
                if p_name in train_cfg:
                    train_cfg[p_name] = p_val
                elif p_name in model_cfg:
                    model_cfg[p_name] = p_val
                else:
                    # Update nested model sub-configs if matching
                    for sub_k in ["attention", "transformer", "classifier"]:
                        if sub_k in model_cfg and isinstance(model_cfg[sub_k], dict):
                            if p_name in model_cfg[sub_k]:
                                model_cfg[sub_k][p_name] = p_val

            merged_best_cfg["model"] = model_cfg
            merged_best_cfg["training"] = train_cfg

            with open(best_config_path, "w", encoding="utf-8") as f:
                yaml.dump(merged_best_cfg, f, default_flow_style=False)

        # Summary JSON
        summary_path = os.path.join(self.output_dir, "summary.json")
        summary_data = {
            "total_trials": len(trials),
            "completed_trials": len([t for t in trials if t.status.name == "COMPLETED"]),
            "failed_trials": len([t for t in trials if t.status.name == "FAILED"]),
            "best_trial_id": best_trial.trial_id if best_trial else None,
            "best_score": best_trial.score if best_trial else None,
            "best_parameters": best_trial.params if best_trial else {},
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

        return {
            "trials_csv": trials_csv_path,
            "leaderboard_csv": leaderboard_csv_path,
            "best_config_yaml": best_config_path,
            "summary_json": summary_path,
        }
