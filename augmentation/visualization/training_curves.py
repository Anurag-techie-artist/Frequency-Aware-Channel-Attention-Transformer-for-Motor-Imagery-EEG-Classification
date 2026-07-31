"""
WGAN-GP Training Curves Visualization Plot.
"""

import os
import matplotlib.pyplot as plt
from typing import List, Dict, Any


def plot_training_curves(
    history: List[Dict[str, Any]],
    save_path: str = None,
) -> plt.Figure:
    """
    Plot Generator loss, Critic loss, and Wasserstein distance curves.

    Args:
        history: List of epoch dictionary records
        save_path: Optional output file path for saving PNG figure

    Returns:
        Matplotlib Figure instance
    """
    epochs = [h["epoch"] for h in history]
    c_losses = [h["critic_loss"] for h in history]
    g_losses = [h["generator_loss"] for h in history]
    w_dists = [h["wasserstein_distance"] for h in history]

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

    axes[0].plot(epochs, c_losses, color="firebrick", label="Critic Loss")
    axes[0].plot(epochs, g_losses, color="darkblue", label="Generator Loss")
    axes[0].set_ylabel("Loss Score")
    axes[0].set_title("Conditional WGAN-GP Training Loss Progression")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, w_dists, color="forestgreen", label="Wasserstein Distance")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("W-Distance")
    axes[1].set_title("Estimated Wasserstein Distance Metric")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
