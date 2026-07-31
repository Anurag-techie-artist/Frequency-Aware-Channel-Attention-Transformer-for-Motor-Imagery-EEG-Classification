"""
Classification Performance Metrics Computation (Accuracy, Precision, Recall, Macro F1).
"""

from typing import Dict, Any
import torch
from metrics.accuracy import compute_accuracy


def compute_classification_metrics(
    logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 4
) -> Dict[str, float]:
    """
    Compute classification metrics dictionary: accuracy, precision, recall, macro f1.

    Args:
        logits: Logits tensor of shape (B, num_classes)
        targets: Target class labels of shape (B,)
        num_classes: Number of target classes

    Returns:
        Dictionary of metric floats
    """
    preds = torch.argmax(logits, dim=1)
    acc = compute_accuracy(logits, targets)

    # Calculate per-class precision, recall, f1
    precisions = []
    recalls = []
    f1s = []

    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum().item()
        fp = ((preds == c) & (targets != c)).sum().item()
        fn = ((preds != c) & (targets == c)).sum().item()

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    macro_f1 = sum(f1s) / num_classes if num_classes > 0 else 0.0
    macro_precision = sum(precisions) / num_classes if num_classes > 0 else 0.0
    macro_recall = sum(recalls) / num_classes if num_classes > 0 else 0.0

    return {
        "accuracy": acc,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1,
    }
