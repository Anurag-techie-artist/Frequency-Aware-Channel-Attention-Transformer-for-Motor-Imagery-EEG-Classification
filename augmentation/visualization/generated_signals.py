"""
Real vs Fake EEG Waveform Visual Comparison.
"""

import os
import matplotlib.pyplot as plt
import torch


def plot_generated_signals(
    real_eeg: torch.Tensor,
    fake_eeg: torch.Tensor,
    channel_idx: int = 0,
    band_idx: int = 0,
    save_path: str = None,
) -> plt.Figure:
    """
    Plot temporal waveform comparison between real and synthetic EEG signals.

    Args:
        real_eeg: Real EEG tensor (B, Bands, Channels, Samples)
        fake_eeg: Fake EEG tensor (B, Bands, Channels, Samples)
        channel_idx: Channel index to plot
        band_idx: Frequency band index to plot
        save_path: Optional output file path for saving PNG figure

    Returns:
        Matplotlib Figure instance
    """
    fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True)

    r_sig = real_eeg[0, band_idx, channel_idx].detach().cpu().numpy()
    f_sig = fake_eeg[0, band_idx, channel_idx].detach().cpu().numpy()

    axes[0].plot(r_sig, color="mediumblue", label="Real EEG")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(f"Real EEG Signal (Band {band_idx}, Channel {channel_idx})")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(f_sig, color="crimson", label="Synthetic WGAN-GP EEG")
    axes[1].set_xlabel("Time Samples")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_title(f"Synthetic EEG Signal (Band {band_idx}, Channel {channel_idx})")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
