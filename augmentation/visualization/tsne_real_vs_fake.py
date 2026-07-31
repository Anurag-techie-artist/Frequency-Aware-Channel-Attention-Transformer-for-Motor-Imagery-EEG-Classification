"""
t-SNE / PCA Real vs Synthetic EEG Distribution Overlay Projection Plot.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA


def plot_tsne_real_vs_fake(
    real_eeg: torch.Tensor,
    fake_eeg: torch.Tensor,
    save_path: str = None,
) -> plt.Figure:
    """
    Plot 2D PCA / t-SNE projection comparing real vs fake EEG distributions.

    Args:
        real_eeg: Real EEG tensor (B_real, Bands, Channels, Samples)
        fake_eeg: Fake EEG tensor (B_fake, Bands, Channels, Samples)
        save_path: Optional output file path for saving PNG figure

    Returns:
        Matplotlib Figure instance
    """
    r_flat = real_eeg.detach().cpu().view(real_eeg.shape[0], -1).numpy()
    f_flat = fake_eeg.detach().cpu().view(fake_eeg.shape[0], -1).numpy()

    combined = np.vstack([r_flat, f_flat])
    pca = PCA(n_components=2)
    proj = pca.fit_transform(combined)

    r_proj = proj[: len(r_flat)]
    f_proj = proj[len(r_flat) :]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(r_proj[:, 0], r_proj[:, 1], color="royalblue", alpha=0.6, label="Real EEG", edgecolors="none")
    ax.scatter(f_proj[:, 0], f_proj[:, 1], color="crimson", alpha=0.6, label="Synthetic WGAN-GP EEG", edgecolors="none")

    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title("Real vs Synthetic EEG Distribution Overlay")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
