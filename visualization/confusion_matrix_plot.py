"""
Confusion Matrix Heatmap Visualization Utility.
"""

import os
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list = None,
    normalize: bool = True,
    title: str = "Confusion Matrix",
    save_path: str = None,
) -> plt.Figure:
    """
    Plot confusion matrix heatmap and optionally save to disk.

    Args:
        cm: Confusion matrix numpy array or list of shape (K, K)
        class_names: List of target class label strings
        normalize: If True, normalizes counts by row sums
        title: Plot title string
        save_path: Optional output file path for saving PNG figure

    Returns:
        Matplotlib Figure instance
    """
    cm_array = np.array(cm, dtype=np.float64)
    if class_names is None:
        class_names = [f"Class {i}" for i in range(cm_array.shape[0])]

    display_cm = (
        cm_array / (cm_array.sum(axis=1, keepdims=True) + 1e-8)
        if normalize
        else cm_array
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(display_cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, label="Proportion" if normalize else "Count")

    ax.set(
        xticks=np.arange(cm_array.shape[1]),
        yticks=np.arange(cm_array.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = ".2f" if normalize else "d"
    thresh = display_cm.max() / 2.0
    for i in range(cm_array.shape[0]):
        for j in range(cm_array.shape[1]):
            val = display_cm[i, j] if normalize else int(cm_array[i, j])
            ax.text(
                j,
                i,
                format(val, fmt),
                ha="center",
                va="center",
                color="white" if display_cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
