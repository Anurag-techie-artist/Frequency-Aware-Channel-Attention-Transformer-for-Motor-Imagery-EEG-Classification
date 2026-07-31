"""
Unit Tests for Visualization Package (Phase 8).
"""

import os
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from visualization import (
    plot_confusion_matrix,
    plot_learning_curves,
    plot_embedding_projection,
    plot_attention_heatmap,
)


class TestVisualization(unittest.TestCase):
    """Test suite for figure generation functions."""

    def test_plot_confusion_matrix(self):
        """Test confusion matrix heatmap generation."""
        cm = np.array([[10, 2], [1, 15]])
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = os.path.join(tmp_dir, "cm.png")
            fig = plot_confusion_matrix(cm, save_path=save_path)
            self.assertIsInstance(fig, plt.Figure)
            self.assertTrue(os.path.exists(save_path))
            plt.close(fig)

    def test_plot_learning_curves(self):
        """Test learning curves plot generation."""
        df = pd.DataFrame({
            "epoch": [1, 2],
            "train_loss": [0.5, 0.3],
            "val_loss": [0.6, 0.4],
            "train_accuracy": [0.7, 0.85],
            "val_accuracy": [0.65, 0.8],
        })
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = os.path.join(tmp_dir, "lc.png")
            fig = plot_learning_curves(df, save_path=save_path)
            self.assertIsInstance(fig, plt.Figure)
            self.assertTrue(os.path.exists(save_path))
            plt.close(fig)

    def test_plot_embedding_projection(self):
        """Test 2D scatter plot generation."""
        emb_2d = np.random.randn(20, 2)
        targets = np.random.randint(0, 4, 20)
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = os.path.join(tmp_dir, "tsne.png")
            fig = plot_embedding_projection(emb_2d, targets, method="t-SNE", save_path=save_path)
            self.assertIsInstance(fig, plt.Figure)
            self.assertTrue(os.path.exists(save_path))
            plt.close(fig)

    def test_plot_attention_heatmap(self):
        """Test ACA channel attention heatmap generation."""
        att = np.random.rand(4, 133)
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = os.path.join(tmp_dir, "att.png")
            fig = plot_attention_heatmap(att, save_path=save_path)
            self.assertIsInstance(fig, plt.Figure)
            self.assertTrue(os.path.exists(save_path))
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
