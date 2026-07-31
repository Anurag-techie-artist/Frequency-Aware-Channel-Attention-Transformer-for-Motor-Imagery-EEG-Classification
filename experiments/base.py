"""
Abstract Base Class for Long-Running Experiments.

Establishes a unified lifecycle contract across Training, Evaluation, Hyperparameter Optimization,
Data Augmentation (WGAN-GP), Benchmarking, and Reproducibility experiments.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseExperiment(ABC):
    """Abstract Base Class for long-running experiment lifecycles."""

    @abstractmethod
    def run(self, **kwargs) -> Any:
        """Execute the experiment from start to finish."""
        pass

    def resume(self, checkpoint_path_or_dir: str, **kwargs) -> Any:
        """Resume an interrupted experiment run."""
        return self.run(resume=True, **kwargs)

    @abstractmethod
    def summarize(self) -> Dict[str, Any]:
        """Summarize experiment results and return metadata dictionary."""
        pass
