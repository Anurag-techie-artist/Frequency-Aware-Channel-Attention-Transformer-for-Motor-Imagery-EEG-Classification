"""
Ablation Study Runner Module.

Executes comparative evaluation across model architectural variants
(Baseline, Frequency Only, + ACA, + FATE, Full Model) and exports ablation_results.csv.
"""

import os
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import torch

from configs.config_loader import load_master_config
from models.eeg_motor_imagery_model import EEGMotorImageryModel
from datasets.builder import build_dataloaders
from evaluation.evaluator import Evaluator

logger = logging.getLogger(__name__)


class AblationRunner:
    """Orchestrates comparative ablation experiments across architectural components."""

    def __init__(
        self,
        train_config_path: Optional[str] = None,
        output_dir: str = "outputs/evaluation",
    ):
        self.output_dir = output_dir
        self.config = load_master_config(train_cfg_path=train_config_path)
        os.makedirs(output_dir, exist_ok=True)

    def run_ablation(self, dataloader: Optional[Any] = None) -> pd.DataFrame:
        """
        Execute ablation study matrix across model variants.

        Args:
            dataloader: Optional DataLoader to use for evaluation. If None, builds from config.

        Returns:
            Pandas DataFrame containing comparative ablation metrics table
        """
        if dataloader is None:
            _, val_loader, test_loader = build_dataloaders(self.config)
            dataloader = test_loader or val_loader

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define ablation model variants
        variants = [
            ("Baseline (Raw EEG)", {"attention": False, "transformer": False}),
            ("Frequency Only", {"attention": False, "transformer": True}),
            ("+ ACA", {"attention": True, "transformer": False}),
            ("+ FATE (Full Model)", {"attention": True, "transformer": True}),
        ]

        results_rows = []

        for variant_name, flags in variants:
            logger.info(f"Evaluating ablation variant: {variant_name}...")

            # Modify config copy for variant
            var_config = dict(self.config)
            model_cfg = dict(var_config.get("model", {}))
            att_cfg = dict(model_cfg.get("attention", {}))
            att_cfg["enabled"] = flags["attention"]
            model_cfg["attention"] = att_cfg
            var_config["model"] = model_cfg

            # Instantiate variant model
            model = EEGMotorImageryModel.from_config(var_config)
            model.eval()

            evaluator = Evaluator(
                model=model,
                output_dir=os.path.join(self.output_dir, "variants", variant_name.replace(" ", "_")),
                device=device,
            )

            metrics, _ = evaluator.evaluate(
                dataloader=dataloader,
                config=var_config,
                checkpoint_name=variant_name,
                generate_plots=False,
            )

            results_rows.append({
                "Variant": variant_name,
                "Frequency": True,
                "ACA": flags["attention"],
                "FATE": flags["transformer"],
                "Accuracy": metrics["accuracy"],
                "Balanced Accuracy": metrics["balanced_accuracy"],
                "Macro F1": metrics["f1"],
                "Cohen Kappa": metrics["cohen_kappa"],
            })

        df = pd.DataFrame(results_rows)
        csv_path = os.path.join(self.output_dir, "ablation_results.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Ablation study completed successfully! Results exported to {csv_path}")
        return df
