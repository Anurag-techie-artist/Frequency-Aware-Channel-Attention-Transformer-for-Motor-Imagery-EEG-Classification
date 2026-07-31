"""
CLS Embedding Projection Scatter Plot Visualization (PCA & t-SNE).
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def plot_embedding_projection(
    embeddings_2d: np.ndarray,
    targets: np.ndarray,
    class_names: list = None,
    method: str = "t-SNE",
    save_path: str = None,
) -> plt.Figure:
    """
    Plot 2D scatter plot of CLS embedding projections colored by ground truth class labels.

    Args:
        embeddings_2d: 2D numpy array of shape (N, 2) from PCA or t-SNE
        targets: Ground truth class index array of shape (N,)
        class_names: List of target class label strings
        method: Dimensionality reduction technique name ("t-SNE" or "PCA")
        save_path: Optional output file path for saving PNG figure

    Returns:
        Matplotlib Figure instance
    """
    unique_classes = np.unique(targets)
    if class_names is None:
        class_names = [f"Class {c}" for c in range(max(unique_classes) + 1)]

    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("tab10")

    for c in unique_classes:
        mask = (targets == c)
        label = class_names[c] if c < len(class_names) else f"Class {c}"
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[cmap(c % 10)],
            label=label,
            alpha=0.8,
            edgecolors="none",
            s=40,
        )

    ax.set_title(f"2D {method} Projection of Global CLS Embeddings")
    ax.set_xlabel(f"{method} Component 1")
    ax.set_ylabel(f"{method} Component 2")
    ax.legend(title="Motor Imagery Class", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
