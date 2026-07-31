"""
Bootstrap Resampling Utility.
"""

import numpy as np
from typing import Tuple, List, Union


def bootstrap_resample(
    values: Union[List[float], np.ndarray],
    num_bootstraps: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Perform bootstrap resampling to compute robust empirical mean and confidence intervals.

    Args:
        values: Vector of empirical observations
        num_bootstraps: Number of bootstrap samples
        confidence: Confidence level
        seed: Random seed

    Returns:
        Tuple of (bootstrap_mean, ci_lower, ci_upper)
    """
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    if len(arr) == 1:
        return float(arr[0]), float(arr[0]), float(arr[0])

    rng = np.random.RandomState(seed)
    boot_means = []

    for _ in range(num_bootstraps):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_means.append(np.mean(sample))

    boot_means = np.array(boot_means)
    alpha = (1.0 - confidence) / 2.0
    lower = float(np.percentile(boot_means, alpha * 100))
    upper = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    mean = float(np.mean(boot_means))

    return mean, lower, upper
