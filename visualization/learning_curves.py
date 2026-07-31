"""
Learning Curves Progression Visualization.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_learning_curves(
    csv_path_or_df,
    save_path: str = None,
) -> plt.Figure:
    """
    Plot training & validation loss and accuracy progression curves over epochs.

    Args:
        csv_path_or_df: Path string to metrics.csv or pandas DataFrame
        save_path: Optional output file path for saving PNG figure

    Returns:
        Matplotlib Figure instance
    """
    if isinstance(csv_path_or_df, str):
        if not os.path.exists(csv_path_or_df):
            raise FileNotFoundError(f"Metrics CSV file not found at {csv_path_or_df}")
        df = pd.read_csv(csv_path_or_df)
    else:
        df = csv_path_or_df

    epochs = df["epoch"] if "epoch" in df.columns else range(1, len(df) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curve
    if "train_loss" in df.columns:
        ax1.plot(epochs, df["train_loss"], label="Train Loss", color="crimson")
    if "val_loss" in df.columns:
        ax1.plot(epochs, df["val_loss"], label="Val Loss", color="royalblue", linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy curve
    if "train_accuracy" in df.columns:
        ax2.plot(epochs, df["train_accuracy"], label="Train Accuracy", color="crimson")
    if "val_accuracy" in df.columns:
        ax2.plot(epochs, df["val_accuracy"], label="Val Accuracy", color="royalblue", linestyle="--")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
