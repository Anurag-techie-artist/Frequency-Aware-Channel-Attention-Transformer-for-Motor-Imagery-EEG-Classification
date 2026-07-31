"""
ACA Channel Attention Heatmap Visualization Utility.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def plot_attention_heatmap(
    attention_weights: np.ndarray,
    band_names: list = None,
    channel_indices: list = None,
    save_path: str = None,
) -> plt.Figure:
    """
    Plot heatmap of ACA channel attention weights (Bands x Channels).

    Args:
        attention_weights: Numpy array of shape (Bands, Channels) or (Batch, Bands, Channels)
        band_names: List of frequency band names
        channel_indices: List of channel index labels
        save_path: Optional output file path for saving PNG figure

    Returns:
        Matplotlib Figure instance
    """
    if attention_weights.ndim == 3:
        weights = attention_weights.mean(axis=0)  # Average over batch
    else:
        weights = attention_weights

    num_bands, num_channels = weights.shape
    if band_names is None:
        band_names = [f"Band {b+1}" for b in range(num_bands)]

    fig, ax = plt.subplots(figsize=(10, 3.5))
    im = ax.imshow(weights, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax, label="ACA Attention Weight w")

    ax.set_yticks(range(num_bands))
    ax.set_yticklabels(band_names)
    ax.set_xlabel("EEG Channel Index")
    ax.set_ylabel("Frequency Band")
    ax.set_title("Adaptive Channel Attention Weight Distribution across Spectral Bands")

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
