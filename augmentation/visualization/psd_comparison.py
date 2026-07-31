"""
Real vs Synthetic Power Spectral Density (PSD) Overlay Curves Visualization.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_psd_comparison(
    real_eeg: torch.Tensor,
    fake_eeg: torch.Tensor,
    save_path: str = None,
) -> plt.Figure:
    """
    Plot overlay curves of real vs synthetic Power Spectral Density.

    Args:
        real_eeg: Real EEG tensor (B, Bands, Channels, Samples)
        fake_eeg: Fake EEG tensor (B, Bands, Channels, Samples)
        save_path: Optional output file path for saving PNG figure

    Returns:
        Matplotlib Figure instance
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    r_psd = torch.mean(real_eeg ** 2, dim=(0, 2)).detach().cpu().numpy()
    f_psd = torch.mean(fake_eeg ** 2, dim=(0, 2)).detach().cpu().numpy()

    num_bands, num_samples = r_psd.shape
    x_axis = np.linspace(0, 100, num_samples)

    for b in range(num_bands):
        ax.plot(x_axis, r_psd[b], "--", label=f"Real Band {b}", alpha=0.7)
        ax.plot(x_axis, f_psd[b], "-", label=f"Synthetic Band {b}", alpha=0.7)

    ax.set_xlabel("Sample Index / Frequency Index")
    ax.set_ylabel("Power Spectral Density")
    ax.set_title("Real vs Synthetic Power Spectral Density Comparison")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
