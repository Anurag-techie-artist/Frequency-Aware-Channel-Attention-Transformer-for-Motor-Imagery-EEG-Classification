"""
Statistical Significance Testing Utility (Paired t-test and Wilcoxon Signed-Rank Test).
"""

import numpy as np
from scipy import stats
from typing import Dict, Any, Union, List


def compute_statistical_significance(
    baseline_scores: Union[List[float], np.ndarray],
    augmented_scores: Union[List[float], np.ndarray],
) -> Dict[str, Any]:
    """
    Compute paired t-test and Wilcoxon signed-rank test p-values.

    Args:
        baseline_scores: Accuracy or metric scores across random seeds/folds for Baseline
        augmented_scores: Metric scores across random seeds/folds for Augmented model

    Returns:
        Dictionary containing p_value_ttest, p_value_wilcoxon, and is_statistically_significant
    """
    b = np.array(baseline_scores, dtype=float)
    a = np.array(augmented_scores, dtype=float)

    if len(b) != len(a):
        raise ValueError(f"Sample length mismatch: baseline ({len(b)}) vs augmented ({len(a)})")

    if len(b) < 2:
        return {
            "t_statistic": 0.0,
            "p_value_ttest": 1.0,
            "p_value_wilcoxon": 1.0,
            "is_statistically_significant": False,
        }

    # Paired Student's t-test
    t_stat, p_ttest = stats.ttest_rel(a, b)

    # Wilcoxon signed-rank test
    diff = a - b
    if np.all(diff == 0):
        p_wilcoxon = 1.0
    else:
        _, p_wilcoxon = stats.wilcoxon(diff)

    return {
        "t_statistic": float(t_stat) if not np.isnan(t_stat) else 0.0,
        "p_value_ttest": float(p_ttest) if not np.isnan(p_ttest) else 1.0,
        "p_value_wilcoxon": float(p_wilcoxon) if not np.isnan(p_wilcoxon) else 1.0,
        "is_statistically_significant": float(p_ttest) < 0.05 if not np.isnan(p_ttest) else False,
    }
