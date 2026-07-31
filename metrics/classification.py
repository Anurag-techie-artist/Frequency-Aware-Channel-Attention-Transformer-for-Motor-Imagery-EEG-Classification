"""
Classification Performance Metrics Computation.

Computes Accuracy, Precision, Recall, Macro/Micro/Weighted F1, Balanced Accuracy,
Cohen's Kappa, and Per-Class Metrics.
"""

from typing import Dict, Any
import torch

from metrics.accuracy import compute_accuracy
from metrics.balanced_accuracy import compute_balanced_accuracy
from metrics.cohen_kappa import compute_cohen_kappa
from metrics.confusion_matrix import compute_confusion_matrix


def compute_classification_metrics(
    logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 4
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics dictionary.

    Args:
        logits: Logits tensor of shape (B, num_classes)
        targets: Target class labels of shape (B,)
        num_classes: Number of target classes

    Returns:
        Dictionary of metric values and per-class breakdowns
    """
    preds = torch.argmax(logits, dim=1) if logits.dim() > 1 else logits
    acc = compute_accuracy(logits, targets)
    bal_acc = compute_balanced_accuracy(logits, targets, num_classes=num_classes)
    kappa = compute_cohen_kappa(logits, targets, num_classes=num_classes)
    cm = compute_confusion_matrix(logits, targets, num_classes=num_classes)

    precisions = []
    recalls = []
    f1s = []
    support = []

    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum().item()
        fp = ((preds == c) & (targets != c)).sum().item()
        fn = ((preds != c) & (targets == c)).sum().item()
        sup = (targets == c).sum().item()

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        support.append(sup)

    total_samples = sum(support)
    macro_f1 = float(sum(f1s) / num_classes) if num_classes > 0 else 0.0
    macro_precision = float(sum(precisions) / num_classes) if num_classes > 0 else 0.0
    macro_recall = float(sum(recalls) / num_classes) if num_classes > 0 else 0.0

    # Weighted F1
    weighted_f1 = (
        sum(f1 * sup for f1, sup in zip(f1s, support)) / total_samples
        if total_samples > 0
        else 0.0
    )

    per_class = {
        f"class_{c}": {
            "precision": precisions[c],
            "recall": recalls[c],
            "f1": f1s[c],
            "support": support[c],
        }
        for c in range(num_classes)
    }

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "cohen_kappa": float(kappa),
        "precision": float(macro_precision),
        "recall": float(macro_recall),
        "f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }
