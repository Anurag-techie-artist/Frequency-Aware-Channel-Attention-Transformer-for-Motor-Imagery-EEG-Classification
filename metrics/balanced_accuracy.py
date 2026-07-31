"""
Balanced Accuracy Metric Computation.

Calculates the average of recall obtained on each class.
"""

import torch


def compute_balanced_accuracy(logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 4) -> float:
    """
    Compute balanced accuracy ratio in range [0.0, 1.0].

    Args:
        logits: Logits tensor of shape (B, num_classes) or predicted class indices (B,)
        targets: Ground truth class index labels of shape (B,)
        num_classes: Total target class count

    Returns:
        Balanced accuracy float ratio
    """
    if logits.dim() > 1 and logits.size(1) > 1:
        preds = torch.argmax(logits, dim=1)
    else:
        preds = logits

    recalls = []
    for c in range(num_classes):
        mask = (targets == c)
        total_c = mask.sum().item()
        if total_c == 0:
            continue
        correct_c = ((preds == c) & mask).sum().item()
        recalls.append(correct_c / total_c)

    return float(sum(recalls) / len(recalls)) if len(recalls) > 0 else 0.0
