"""
Unified ArtifactManager Module for Phase 10 Augmentation Framework.

Provides standardized API for checkpoints, synthetic dataset export, manifests,
evaluation reports, and statistical result summaries.
"""

import os
import json
import time
import torch
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from evaluation.manifest import get_git_commit_hash, compute_config_hash


class ArtifactManager:
    """Unified artifact manager enforcing standardized directory layout and metadata tracing."""

    def __init__(self, output_dir: str = "outputs/augmentation"):
        self.output_dir = output_dir
        self.gan_dir = os.path.join(output_dir, "gan")
        self.synthetic_dir = os.path.join(output_dir, "synthetic")
        self.eval_dir = os.path.join(output_dir, "evaluation")
        self.manifest_dir = os.path.join(output_dir, "manifests")

        os.makedirs(self.gan_dir, exist_ok=True)
        os.makedirs(self.synthetic_dir, exist_ok=True)
        os.makedirs(self.eval_dir, exist_ok=True)
        os.makedirs(self.manifest_dir, exist_ok=True)

    def save_synthetic_dataset(
        self,
        synthetic_x: torch.Tensor,
        synthetic_y: torch.Tensor,
        metadata: Dict[str, Any],
    ) -> Tuple[str, str]:
        """Save synthetic dataset tensor and metadata JSON."""
        data_path = os.path.join(self.synthetic_dir, "generated_dataset.pt")
        meta_path = os.path.join(self.synthetic_dir, "metadata.json")

        torch.save({"x": synthetic_x, "y": synthetic_y}, data_path)

        full_meta = dict(metadata)
        full_meta["samples"] = synthetic_x.shape[0]
        full_meta["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        full_meta["git_commit"] = get_git_commit_hash()

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(full_meta, f, indent=2)

        return data_path, meta_path

    def save_evaluation_report(
        self,
        report_data: Dict[str, Any],
        stats_data: Dict[str, Any],
    ) -> Tuple[str, str]:
        """Save evaluation report JSON and statistics JSON."""
        report_path = os.path.join(self.eval_dir, "report.json")
        stats_path = os.path.join(self.eval_dir, "statistics.json")

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats_data, f, indent=2)

        return report_path, stats_path

    def save_manifests(
        self,
        config: Dict[str, Any],
        generator_params: Dict[str, Any],
    ) -> Tuple[str, str]:
        """Save comprehensive experiment manifest and generator manifest."""
        exp_manifest = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "git_commit": get_git_commit_hash(),
            "config_hash": compute_config_hash(config),
            "config": config,
        }

        exp_path = os.path.join(self.manifest_dir, "experiment.json")
        gen_path = os.path.join(self.manifest_dir, "generator.json")

        with open(exp_path, "w", encoding="utf-8") as f:
            json.dump(exp_manifest, f, indent=2)

        with open(gen_path, "w", encoding="utf-8") as f:
            json.dump(generator_params, f, indent=2)

        return exp_path, gen_path
