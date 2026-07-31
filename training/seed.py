"""
Global Reproducibility Seed Utility.
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> int:
    """
    Set global random seeds across Python, NumPy, PyTorch CPU & CUDA for reproducibility.

    Args:
        seed: Integer seed value

    Returns:
        Configured seed integer
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def set_global_seed(seed: int = 42) -> int:
    """Alias for set_seed for global reproducibility manager contract."""
    return set_seed(seed)
