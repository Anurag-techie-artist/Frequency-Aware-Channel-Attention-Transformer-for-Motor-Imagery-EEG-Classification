"""
Optimization History Progression Plot Visualization.
"""

import os
import matplotlib.pyplot as plt
import pandas as pd


def plot_optimization_history(
    trials_csv_or_df,
    metric_name: str = "val_accuracy",
    save_path: str = None,
) -> plt.Figure:
    """
    Plot optimization metric score progression over trial sequence.

    Args:
        trials_csv_or_df: Path string to trials.csv or pandas DataFrame
        metric_name: Metric label string
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

    fig, ax = plt.subplots(figsize=(8, 4))
    if not df.empty and "score" in df.columns:
        scores = df["score"].values
        trial_ids = df["trial_id"].values if "trial_id" in df.columns else range(len(scores))

        # Best trajectory line
        best_so_far = []
        curr_best = -float("inf")
        for s in scores:
            if not pd.isna(s) and s > curr_best:
                curr_best = s
            best_so_far.append(curr_best)

        ax.plot(trial_ids, scores, "o-", color="cornflowerblue", alpha=0.7, label="Trial Score")
        ax.plot(trial_ids, best_so_far, "s--", color="crimson", label="Best Score")

    ax.set_xlabel("Trial ID")
    ax.set_ylabel(metric_name)
    ax.set_title(f"HPO Optimization History Curve ({metric_name})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
