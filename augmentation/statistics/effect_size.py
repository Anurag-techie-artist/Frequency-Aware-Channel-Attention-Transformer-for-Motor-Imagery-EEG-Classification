"""
Effect Size Calculations (Cohen's d and Hedges' g).
"""

import numpy as np
from typing import Dict, Any, Union, List


def compute_effect_size(
    baseline_scores: Union[List[float], np.ndarray],
    augmented_scores: Union[List[float], np.ndarray],
) -> Dict[str, float]:
    """
    Compute Cohen's d and Hedges' g effect sizes.

    Args:
        baseline_scores: Baseline metric values
        augmented_scores: Augmented metric values

    Returns:
        Dictionary containing cohens_d and hedges_g
    """
    b = np.array(baseline_scores, dtype=float)
    a = np.array(augmented_scores, dtype=float)

    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return {"cohens_d": 0.0, "hedges_g": 0.0}

    s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
    s_pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))

    if s_pooled == 0 or np.isnan(s_pooled):
        return {"cohens_d": 0.0, "hedges_g": 0.0}

    d = (np.mean(a) - np.mean(b)) / s_pooled

    # Hedges' g correction factor
    j = 1.0 - (3.0 / (4.0 * (n1 + n2) - 9.0))
    g = d * j

    return {
        "cohens_d": float(d),
        "hedges_g": float(g),
    }
