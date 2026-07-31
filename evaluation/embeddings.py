"""
Embedding Dimensionality Reduction Projector (PCA & t-SNE).
"""

import numpy as np


class EmbeddingProjector:
    """Projects high-dimensional CLS embeddings (B, d_model) to 2D coordinates for visualization."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def project_pca(self, embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
        """
        Reduce dimensions using Principal Component Analysis (PCA).

        Args:
            embeddings: High-dimensional array of shape (N, d_model)
            n_components: Target dimensions (default 2)

        Returns:
            Projected array of shape (N, n_components)
        """
        # Center the data
        mean = np.mean(embeddings, axis=0)
        centered = embeddings - mean

        # Singular Value Decomposition (SVD)
        u, s, vt = np.linalg.svd(centered, full_matrices=False)
        projected = np.dot(centered, vt[:n_components].T)
        return projected

    def project_tsne(self, embeddings: np.ndarray, perplexity: float = 30.0) -> np.ndarray:
        """
        Reduce dimensions using t-Distributed Stochastic Neighbor Embedding (t-SNE).

        Args:
            embeddings: High-dimensional array of shape (N, d_model)
            perplexity: t-SNE perplexity parameter

        Returns:
            Projected array of shape (N, 2)
        """
        try:
            from sklearn.manifold import TSNE

            perp = min(perplexity, max(1.0, float(embeddings.shape[0] - 1)))
            tsne = TSNE(
                n_components=2,
                perplexity=perp,
                random_state=self.seed,
                init="pca",
            )
            return tsne.fit_transform(embeddings)
        except ImportError:
            # Fallback to PCA if scikit-learn is unavailable
            return self.project_pca(embeddings, n_components=2)
