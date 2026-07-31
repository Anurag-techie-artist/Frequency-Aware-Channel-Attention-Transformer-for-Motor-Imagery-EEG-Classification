"""
ReportGenerator Module for Exporting Scientific Evaluation Reports.

Exports evaluation_report.json, metrics.csv, predictions.csv, and raw_predictions.pt.
"""

import os
import json
import csv
from typing import Dict, Any, Optional

import torch
import pandas as pd

from evaluation.inference import InferenceResults


class ReportGenerator:
    """Exports structured evaluation metrics, prediction tables, and raw inference tensors."""

    def __init__(self, output_dir: str = "outputs/evaluation"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def export_report_json(self, metrics: Dict[str, Any], filename: str = "evaluation_report.json") -> str:
        """Export metrics dictionary as indented JSON file."""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        return filepath

    def export_metrics_csv(self, metrics: Dict[str, Any], filename: str = "metrics.csv") -> str:
        """Export high-level metrics as CSV table."""
        filepath = os.path.join(self.output_dir, filename)
        flat_metrics = {
            k: v for k, v in metrics.items() if not isinstance(v, (dict, list))
        }
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat_metrics.keys()))
            writer.writeheader()
            writer.writerow(flat_metrics)
        return filepath

    def export_predictions_csv(
        self, results: InferenceResults, class_names: Optional[list] = None, filename: str = "predictions.csv"
    ) -> str:
        """Export sample-level predictions and Softmax probabilities to CSV file."""
        filepath = os.path.join(self.output_dir, filename)
        probs_np = results.probabilities.numpy()
        preds_np = results.predicted_class.numpy()
        targets_np = results.targets.numpy()

        num_samples = len(preds_np)
        num_classes = probs_np.shape[1] if probs_np.ndim > 1 else 1

        if class_names is None:
            class_names = [f"Class_{c}" for c in range(num_classes)]

        rows = []
        for i in range(num_samples):
            row = {
                "Sample_ID": i,
                "Target_Class": int(targets_np[i]),
                "Predicted_Class": int(preds_np[i]),
                "Correct": bool(targets_np[i] == preds_np[i]),
            }
            for c in range(num_classes):
                col_name = f"Prob_{class_names[c]}" if c < len(class_names) else f"Prob_{c}"
                row[col_name] = float(probs_np[i, c])
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        return filepath

    def export_raw_predictions(self, results: InferenceResults, filename: str = "raw_predictions.pt") -> str:
        """Serialize raw inference tensors to PyTorch binary checkpoint file."""
        filepath = os.path.join(self.output_dir, filename)
        raw_dict = {
            "logits": results.logits,
            "probabilities": results.probabilities,
            "predicted_class": results.predicted_class,
            "targets": results.targets,
            "cls_embeddings": results.cls_embeddings,
            "attention_weights": results.attention_weights,
        }
        torch.save(raw_dict, filepath)
        return filepath
