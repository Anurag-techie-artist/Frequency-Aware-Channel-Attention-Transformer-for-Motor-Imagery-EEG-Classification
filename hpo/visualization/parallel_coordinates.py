"""
Parallel Coordinates Plot Visualization for Hyperparameter Interactions.
"""

import os
import matplotlib.pyplot as plt
import pandas as pd


def plot_parallel_coordinates(
    trials_csv_or_df,
    save_path: str = None,
) -> plt.Figure:
    """
    Plot parallel coordinates graph showing hyperparameter combinations colored by trial score.

    Args:
        trials_csv_or_df: Path string to trials.csv or pandas DataFrame
        save_path: Optional output file path for saving PNG figure

    Returns:
        Matplotlib Figure instance
    """
    if isinstance(trials_csv_or_df, str):
        if not os.path.exists(trials_csv_or_df):
            raise FileNotFoundError(f"Trials CSV file not found at {trials_csv_or_df}")
        df = pd.read_csv(trials_csv_or_df)
    else:
        df = trials_csv_or_df

    # Extract parameter columns
    param_cols = [col for col in df.columns if col.startswith("param_")]
    if not param_cols or "score" not in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Insufficient trial data for Parallel Coordinates", ha="center", va="center")
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
        return fig

    # Normalize numeric columns to [0, 1] for visualization overlay
    df_norm = df[param_cols + ["score"]].copy().dropna()
    for col in param_cols:
        if df_norm[col].dtype == object:
            # Map categories to integers
            df_norm[col] = df_norm[col].astype("category").cat.codes

        c_min = df_norm[col].min()
        c_max = df_norm[col].max()
        if c_max > c_min:
            df_norm[col] = (df_norm[col] - c_min) / (c_max - c_min)
        else:
            df_norm[col] = 0.5

    fig, ax = plt.subplots(figsize=(10, 5))
    x_coords = range(len(param_cols))
    clean_labels = [c.replace("param_", "") for c in param_cols]

    # Color lines by target trial score
    cmap = plt.get_cmap("viridis")
    scores = df_norm["score"].values
    s_min, s_max = scores.min(), scores.max()
    s_range = (s_max - s_min) if s_max > s_min else 1.0

    for idx, row in df_norm.iterrows():
        y_vals = [row[c] for c in param_cols]
        color = cmap((row["score"] - s_min) / s_range)
        ax.plot(x_coords, y_vals, color=color, alpha=0.6, linewidth=1.5)

    ax.set_xticks(x_coords)
    ax.set_xticklabels(clean_labels, rotation=30, ha="right")
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(["Min Bound", "Mid Bound", "Max Bound"])
    ax.set_title("Parallel Coordinates Hyperparameter Interaction Plot")

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=s_min, vmax=s_max))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Trial Validation Score")

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
