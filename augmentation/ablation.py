"""
Augmentation Strategy & Ratio Ablation Study Runner.

Executes ratio sweep (0%, 25%, 50%, 75%, 100%) and strategy comparison (Baseline vs MixUp vs CutMix vs WGAN-GP),
computing 95% CIs, p-values, and Cohen's d effect sizes.
"""

import os
import logging
import pandas as pd
from typing import Dict, Any, List

import torch
from torch.utils.data import DataLoader, TensorDataset

from augmentation.factory import build_augmentation_strategy
from augmentation.pipeline import AugmentationPipeline
from augmentation.statistics import (
    compute_confidence_interval,
    compute_statistical_significance,
    compute_effect_size,
)
from models.eeg_motor_imagery_model import EEGMotorImageryModel
from datasets.builder import build_dataloaders
from training import build_loss, build_optimizer, build_scheduler, Trainer, get_device
from evaluation.evaluator import Evaluator

logger = logging.getLogger(__name__)


class AugmentationRatioAblationRunner:
    """Orchestrates strategy and ratio ablation experiments."""

    def __init__(self, master_config: Dict[str, Any]):
        self.master_config = master_config
        self.output_dir = master_config.get("output", {}).get("output_dir", "outputs/augmentation")
        os.makedirs(self.output_dir, exist_ok=True)

    def run_ablation(
        self,
        ratios: List[float] = [0.0, 0.25, 0.50, 0.75, 1.0],
        strategies: List[str] = ["none", "wgan_gp"],
        seeds: List[int] = [42, 123, 999],
    ) -> pd.DataFrame:
        """
        Execute ratio and strategy ablation sweep across multiple random seeds (RQ1-RQ6).

        Args:
            ratios: List of synthetic augmentation ratios
            strategies: List of augmentation strategy keys
            seeds: Random seed values for multi-seed robustness analysis

        Returns:
            Pandas DataFrame containing leaderboard ablation table
        """
        logger.info(f"Starting Augmentation Ratio Ablation Sweep across ratios={ratios}, strategies={strategies}, seeds={seeds}...")
        records = []
        train_loader, val_loader, _ = build_dataloaders(self.master_config)

        # Baseline evaluation (Ratio 0.0) across seeds
        baseline_scores = []

        for strat_name in strategies:
            for ratio in ratios:
                if strat_name == "none" and ratio > 0.0:
                    continue

                seed_accs = []
                for s in seeds:
                    cfg = dict(self.master_config)
                    cfg["augmentation"] = {"strategy": strat_name, "ratio": ratio, "seed": s}
                    cfg["training"]["epochs"] = 1  # Fast evaluation epoch count for ablation

                    strat = build_augmentation_strategy(cfg)
                    strat.fit(train_loader, cfg)

                    pipeline = AugmentationPipeline(strat)
                    aug_loader = pipeline.augment_dataloader(train_loader, ratio=ratio, batch_size=cfg["training"]["batch_size"])

                    # Train classifier
                    model = EEGMotorImageryModel.from_config(cfg)
                    criterion = build_loss(cfg)
                    optimizer = build_optimizer(model, cfg)
                    scheduler = build_scheduler(optimizer, cfg)
                    device = get_device("auto")

                    trainer = Trainer(model, criterion, optimizer, scheduler, cfg, device=device)
                    trainer.fit(aug_loader, val_loader)

                    val_metrics = trainer.validate_epoch(val_loader)
                    acc = float(val_metrics.get("accuracy", 0.0))
                    seed_accs.append(acc)

                mean_acc, ci_low, ci_high = compute_confidence_interval(seed_accs)
                if strat_name == "none" and ratio == 0.0:
                    baseline_scores = list(seed_accs)

                rec = {
                    "strategy": strat_name,
                    "ratio": ratio,
                    "mean_accuracy": round(mean_acc, 4),
                    "ci_95_lower": round(ci_low, 4),
                    "ci_95_upper": round(ci_high, 4),
                }

                if baseline_scores and len(seed_accs) == len(baseline_scores):
                    sig = compute_statistical_significance(baseline_scores, seed_accs)
                    eff = compute_effect_size(baseline_scores, seed_accs)
                    rec["p_value_ttest"] = round(sig["p_value_ttest"], 4)
                    rec["cohens_d"] = round(eff["cohens_d"], 4)

                records.append(rec)
                logger.info(f"Strategy: {strat_name:<10} | Ratio: {ratio:.2f} | Mean Acc: {mean_acc:.4f} [95% CI: {ci_low:.4f} - {ci_high:.4f}]")

        df_ablation = pd.DataFrame(records)
        df_ablation.sort_values(by="mean_accuracy", ascending=False, inplace=True)
        csv_path = os.path.join(self.output_dir, "leaderboard.csv")
        df_ablation.to_csv(csv_path, index=False)
        return df_ablation
