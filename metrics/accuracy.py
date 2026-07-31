"""
Accuracy Metric Computation Functions.
"""

import torch


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute classification accuracy ratio in range [0.0, 1.0].

    Args:
        logits: Logits or probabilities of shape (B, num_classes)
        targets: Target class index labels of shape (B,)

    Returns:
        Accuracy float ratio
    """
    if logits.dim() > 1 and logits.size(1) > 1:
        preds = torch.argmax(logits, dim=1)
    else:
        preds = logits

    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / total if total > 0 else 0.0
