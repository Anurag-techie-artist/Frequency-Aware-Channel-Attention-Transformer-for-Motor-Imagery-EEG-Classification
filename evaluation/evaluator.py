"""
Evaluator Module for Executing End-to-End Evaluation.

Coordinates checkpoint loading, inference execution, metrics computation, figure generation,
and report exports.
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evaluation.inference import InferenceEngine, InferenceResults
from evaluation.manifest import generate_manifest
from evaluation.report import ReportGenerator
from evaluation.embeddings import EmbeddingProjector
from metrics import compute_classification_metrics
from visualization import (
    plot_confusion_matrix,
    plot_learning_curves,
    plot_embedding_projection,
    plot_attention_heatmap,
)

logger = logging.getLogger(__name__)


class Evaluator:
    """Orchestrates checkpoint evaluation, metric computation, visualization, and report generation."""

    def __init__(
        self,
        model: nn.Module,
        output_dir: str = "outputs/evaluation",
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.output_dir = output_dir
        self.device = device or torch.device("cpu")
        self.engine = InferenceEngine(model=self.model, device=self.device)
        self.reporter = ReportGenerator(output_dir=output_dir)

        os.makedirs(output_dir, exist_ok=True)

    def evaluate(
        self,
        dataloader: DataLoader,
        config: Optional[Dict[str, Any]] = None,
        checkpoint_name: str = "best.pt",
        class_names: Optional[list] = None,
        generate_plots: bool = True,
    ) -> Tuple[Dict[str, Any], InferenceResults]:
        """
        Execute full evaluation pipeline on given DataLoader.

        Args:
            dataloader: PyTorch DataLoader yielding (x_batch, y_batch)
            config: Master configuration dictionary
            checkpoint_name: Checkpoint identifier string
            class_names: Target class names list
            generate_plots: If True, generates PNG visualization plots

        Returns:
            Tuple of (metrics_dict, inference_results)
        """
        if class_names is None:
            class_names = ["Left Hand", "Right Hand", "Both Hands", "Feet"]

        logger.info("Generating evaluation manifest...")
        generate_manifest(
            checkpoint_name=checkpoint_name,
            config=config,
            output_dir=self.output_dir,
        )

        logger.info("Executing model inference engine...")
        results = self.engine.run_inference(dataloader)

        logger.info("Computing classification metrics...")
        metrics = compute_classification_metrics(
            logits=results.logits,
            targets=results.targets,
            num_classes=len(class_names),
        )

        logger.info("Exporting evaluation reports and raw prediction tensors...")
        self.reporter.export_report_json(metrics)
        self.reporter.export_metrics_csv(metrics)
        self.reporter.export_predictions_csv(results, class_names=class_names)
        self.reporter.export_raw_predictions(results)

        if generate_plots:
            logger.info("Generating visual evaluation figures...")
            cm_np = results.logits.argmax(dim=-1).numpy()
            plot_confusion_matrix(
                cm=metrics["confusion_matrix"],
                class_names=class_names,
                save_path=os.path.join(self.output_dir, "confusion_matrix.png"),
            )

            # Generate t-SNE / PCA CLS embedding projections
            if results.cls_embeddings is not None and results.cls_embeddings.shape[0] > 1:
                projector = EmbeddingProjector()
                proj_2d = projector.project_tsne(results.cls_embeddings.numpy())
                plot_embedding_projection(
                    embeddings_2d=proj_2d,
                    targets=results.targets.numpy(),
                    class_names=class_names,
                    method="t-SNE",
                    save_path=os.path.join(self.output_dir, "cls_embeddings_tsne.png"),
                )

            # Generate Learning Curves if metrics.csv exists in outputs/logs
            logs_csv = os.path.join(os.path.dirname(self.output_dir), "logs", "metrics.csv")
            if os.path.exists(logs_csv):
                plot_learning_curves(
                    csv_path_or_df=logs_csv,
                    save_path=os.path.join(self.output_dir, "learning_curves.png"),
                )

            # Generate ACA attention heatmaps if available
            if results.attention_weights is not None:
                plot_attention_heatmap(
                    attention_weights=results.attention_weights.numpy(),
                    save_path=os.path.join(self.output_dir, "attention_heatmap.png"),
                )

        logger.info(
            f"Evaluation completed successfully! Accuracy: {metrics['accuracy']:.4f} | Macro F1: {metrics['f1']:.4f}"
        )
        return metrics, results
