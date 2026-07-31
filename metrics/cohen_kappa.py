"""
Cohen's Kappa Statistic Metric Computation.

Measures inter-rater agreement for categorical items.
"""

import torch


def compute_cohen_kappa(logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 4) -> float:
    """
    Compute Cohen's Kappa coefficient in range [-1.0, 1.0].

    Args:
        logits: Logits tensor of shape (B, num_classes) or predicted class indices (B,)
        targets: Ground truth class index labels of shape (B,)
        num_classes: Total target class count

    Returns:
        Cohen's Kappa float coefficient
    """
    if logits.dim() > 1 and logits.size(1) > 1:
        preds = torch.argmax(logits, dim=1)
    else:
        preds = logits

    n = targets.numel()
    if n == 0:
        return 0.0

    po = (preds == targets).sum().item() / n  # Observed agreement

    pe = 0.0  # Expected agreement by chance
    for c in range(num_classes):
        pred_c = (preds == c).sum().item()
        target_c = (targets == c).sum().item()
        pe += (pred_c / n) * (target_c / n)

    if pe == 1.0:
        return 1.0

    kappa = (po - pe) / (1.0 - pe)
    return float(kappa)
