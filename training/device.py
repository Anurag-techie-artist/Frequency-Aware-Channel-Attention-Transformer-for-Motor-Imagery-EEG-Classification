"""
Device Resolution Utility.
"""

import logging
import torch

logger = logging.getLogger(__name__)


def get_device(preference: str = "auto") -> torch.device:
    """
    Resolve target PyTorch execution device (CUDA, MPS, CPU).

    Args:
        preference: "auto", "cuda", "mps", or "cpu"

    Returns:
        torch.device instance
    """
    pref = preference.lower()

    if pref == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")

    if pref == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        logger.warning("MPS requested but not available. Falling back to CPU.")
        return torch.device("cpu")

    if pref == "cpu":
        return torch.device("cpu")

    # Auto resolution preference: CUDA -> MPS -> CPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")
