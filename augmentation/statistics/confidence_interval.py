"""
Statistical 95% Confidence Interval Calculation.
"""

import numpy as np
from scipy import stats
from typing import Tuple, List, Union


def compute_confidence_interval(
    values: Union[List[float], np.ndarray],
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Compute mean and 95% confidence interval for a list/array of values.

    Args:
        values: Data vector
        confidence: Target confidence level (default 0.95)

    Returns:
        Tuple of (mean, ci_lower, ci_upper)
    """
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    if len(arr) == 1:
        return float(arr[0]), float(arr[0]), float(arr[0])

    mean = float(np.mean(arr))
    sem = stats.sem(arr)
    if sem == 0 or np.isnan(sem):
        return mean, mean, mean

    h = sem * stats.t.ppf((1 + confidence) / 2.0, len(arr) - 1)
    return mean, float(mean - h), float(mean + h)
