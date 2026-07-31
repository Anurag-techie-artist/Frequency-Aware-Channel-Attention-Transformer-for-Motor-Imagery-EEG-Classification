"""
Statistical Summary & Comparison Utilities.

Computes mean, standard deviation, standard error, and confidence intervals across
evaluations or cross-validation folds.
"""

import math
from typing import Dict, Any, List
import numpy as np


def compute_summary_statistics(
    metric_values: List[float], confidence: float = 0.95
) -> Dict[str, float]:
    """
    Compute mean, std_dev, std_err, and confidence interval bounds for a list of metric values.

    Args:
        metric_values: List of metric floats
        confidence: Confidence level float (e.g. 0.95)

    Returns:
        Dictionary containing mean, std_dev, std_err, ci_lower, ci_upper
    """
    if not metric_values:
        return {"mean": 0.0, "std_dev": 0.0, "std_err": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    arr = np.array(metric_values, dtype=np.float64)
    n = len(arr)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    std_err = float(std_val / math.sqrt(n)) if n > 0 else 0.0

    # 95% Z-score margin multiplier approx 1.96
    z_margin = 1.96 if confidence == 0.95 else 2.58
    margin = z_margin * std_err

    return {
        "mean": mean_val,
        "std_dev": std_val,
        "std_err": std_err,
        "ci_lower": mean_val - margin,
        "ci_upper": mean_val + margin,
    }
