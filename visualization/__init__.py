"""
Visualization Package for EEG Model Evaluation.
"""

from visualization.confusion_matrix_plot import plot_confusion_matrix
from visualization.learning_curves import plot_learning_curves
from visualization.embedding_projection import plot_embedding_projection
from visualization.attention_maps import plot_attention_heatmap

__all__ = [
    "plot_confusion_matrix",
    "plot_learning_curves",
    "plot_embedding_projection",
    "plot_attention_heatmap",
]
