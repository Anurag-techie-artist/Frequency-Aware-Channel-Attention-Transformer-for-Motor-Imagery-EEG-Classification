"""
Confusion Matrix Metric Computation.
"""

import torch


def compute_confusion_matrix(
    logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 4
) -> torch.Tensor:
    """
    Compute K x K confusion matrix tensor where cm[i, j] is count of target i predicted as j.

    Args:
        logits: Logits tensor of shape (B, num_classes) or predicted class indices (B,)
        targets: Ground truth class index labels of shape (B,)
        num_classes: Total target class count

    Returns:
        Confusion matrix LongTensor of shape (num_classes, num_classes)
    """
    if logits.dim() > 1 and logits.size(1) > 1:
        preds = torch.argmax(logits, dim=1)
    else:
        preds = logits

    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(targets.view(-1), preds.view(-1)):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t.item(), p.item()] += 1

    return cm
