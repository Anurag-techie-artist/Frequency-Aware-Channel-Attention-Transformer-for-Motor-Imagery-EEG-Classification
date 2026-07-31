"""
AugmentationExperimentRunner Orchestrator Module.

Implements BaseExperiment unified lifecycle contract.
"""

import os
import json
import logging
from typing import Dict, Any, Optional

import torch
from configs.config_loader import load_master_config
from experiments.base import BaseExperiment
from augmentation.factory import build_augmentation_strategy
from augmentation.pipeline import AugmentationPipeline
from augmentation.validator import SyntheticDataValidator
from augmentation.artifacts import ArtifactManager
from augmentation.metrics import PSDSimilarity, BandPowerSimilarity, CovarianceDistance, DiversityScore
from augmentation.statistics import compute_confidence_interval, compute_statistical_significance, compute_effect_size
from augmentation.visualization import plot_generated_signals, plot_psd_comparison, plot_tsne_real_vs_fake, plot_training_curves
from models.eeg_motor_imagery_model import EEGMotorImageryModel
from datasets.builder import build_dataloaders
from training import build_loss, build_optimizer, build_scheduler, Trainer, get_device, set_global_seed
from evaluation.evaluator import Evaluator

logger = logging.getLogger(__name__)


class AugmentationExperimentRunner(BaseExperiment):
    """Orchestrates end-to-end WGAN-GP EEG augmentation experiment adhering to BaseExperiment interface."""

    def __init__(self, gan_config_path: Optional[str] = None):
        self.config = load_master_config(train_cfg_path=gan_config_path)
        aug_cfg = self.config.get("augmentation", {})
        set_global_seed(int(aug_cfg.get("seed", 42)))

        self.output_dir = self.config.get("output", {}).get("output_dir", "outputs/augmentation")
        self.artifact_manager = ArtifactManager(output_dir=self.output_dir)

    def run(self, resume: bool = False, **kwargs) -> Dict[str, Any]:
        """Execute full GAN training, synthetic generation, validation, plots, and classifier evaluation."""
        logger.info("Launching Phase 10 Data Augmentation Experiment...")

        train_loader, val_loader, _ = build_dataloaders(self.config)
        aug_cfg = self.config.get("augmentation", {})
        ratio = float(aug_cfg.get("ratio", 0.50))

        # 1. Fit Augmentation Strategy (Conditional WGAN-GP)
        strategy = build_augmentation_strategy(self.config)
        strategy.fit(train_loader, self.config)

        # 2. Extract real data and generate synthetic data
        real_x_list, real_y_list = [], []
        for x_b, y_b in train_loader:
            real_x_list.append(x_b)
            real_y_list.append(y_b)
        real_x = torch.cat(real_x_list, dim=0)
        real_y = torch.cat(real_y_list, dim=0)

        num_synth = int(real_x.shape[0] * ratio)
        synthetic_ds = strategy.generate(num_samples=num_synth)
        synth_x = synthetic_ds.get_data()
        synth_y = synthetic_ds.get_labels()

        # 3. Validate Synthetic Data Integrity
        _, bands, channels, samples = real_x.shape
        SyntheticDataValidator.validate_synthetic_dataset(
            synth_x, synth_y, expected_bands=bands, expected_channels=channels, expected_samples=samples
        )
        logger.info(f"[OK] Synthetic dataset validated successfully: shape {synth_x.shape}")

        # 4. Export Synthetic Dataset & Manifests
        self.artifact_manager.save_synthetic_dataset(synth_x, synth_y, synthetic_ds.get_metadata())
        self.artifact_manager.save_manifests(self.config, {"strategy": aug_cfg.get("strategy", "wgan_gp")})

        # 5. Compute Standardized GAN Quality Metrics
        psd_sim = PSDSimilarity().compute(real_x, synth_x)
        bp_sim = BandPowerSimilarity().compute(real_x, synth_x)
        cov_dist = CovarianceDistance().compute(real_x, synth_x)
        div_score = DiversityScore().compute(real_x, synth_x)

        eval_report = {
            "psd_similarity": psd_sim,
            "bandpower_similarity": bp_sim,
            "covariance_distance": cov_dist,
            "diversity_score": div_score,
        }

        # 6. Generate Plots
        eval_folder = os.path.join(self.output_dir, "evaluation")
        plot_generated_signals(real_x, synth_x, save_path=os.path.join(eval_folder, "generated_signals.png"))
        plot_psd_comparison(real_x, synth_x, save_path=os.path.join(eval_folder, "psd_comparison.png"))
        plot_tsne_real_vs_fake(real_x, synth_x, save_path=os.path.join(eval_folder, "tsne_real_vs_fake.png"))

        # 7. Train Baseline vs Augmented Classifiers
        device = get_device("auto")

        # Baseline Classifier
        base_model = EEGMotorImageryModel.from_config(self.config)
        base_trainer = Trainer(
            base_model, build_loss(self.config), build_optimizer(base_model, self.config),
            build_scheduler(build_optimizer(base_model, self.config), self.config), self.config, device=device
        )
        base_trainer.fit(train_loader, val_loader)
        base_metrics = base_trainer.validate_epoch(val_loader)

        # Augmented Classifier
        pipeline = AugmentationPipeline(strategy)
        aug_loader = pipeline.augment_dataloader(train_loader, ratio=ratio, batch_size=self.config.get("training", {}).get("batch_size", 32))

        aug_model = EEGMotorImageryModel.from_config(self.config)
        aug_trainer = Trainer(
            aug_model, build_loss(self.config), build_optimizer(aug_model, self.config),
            build_scheduler(build_optimizer(aug_model, self.config), self.config), self.config, device=device
        )
        aug_trainer.fit(aug_loader, val_loader)
        aug_metrics = aug_trainer.validate_epoch(val_loader)

        # Statistical analysis
        b_acc = [base_metrics.get("accuracy", 0.0)]
        a_acc = [aug_metrics.get("accuracy", 0.0)]
        stats_summary = {
            "baseline_accuracy": base_metrics.get("accuracy", 0.0),
            "augmented_accuracy": aug_metrics.get("accuracy", 0.0),
            "effect_size": compute_effect_size(b_acc, a_acc),
            "significance": compute_statistical_significance(b_acc, a_acc),
        }

        self.artifact_manager.save_evaluation_report(eval_report, stats_summary)

        logger.info(
            f"Augmentation Experiment Complete! Baseline Acc: {base_metrics.get('accuracy', 0.0):.4f} | "
            f"Augmented Acc: {aug_metrics.get('accuracy', 0.0):.4f}"
        )
        return eval_report

    def resume(self, checkpoint_path_or_dir: str = None, **kwargs) -> Dict[str, Any]:
        """Resume augmentation experiment."""
        return self.run(resume=True, **kwargs)

    def summarize(self) -> Dict[str, Any]:
        """Summarize experiment results."""
        report_path = os.path.join(self.output_dir, "evaluation", "report.json")
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"status": "no_summary_available"}
