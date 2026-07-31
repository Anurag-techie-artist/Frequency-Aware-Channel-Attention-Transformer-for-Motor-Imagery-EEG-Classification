"""
Statistical Analysis Package.
"""

from augmentation.statistics.confidence_interval import compute_confidence_interval
from augmentation.statistics.significance import compute_statistical_significance
from augmentation.statistics.effect_size import compute_effect_size
from augmentation.statistics.bootstrap import bootstrap_resample

__all__ = [
    "compute_confidence_interval",
    "compute_statistical_significance",
    "compute_effect_size",
    "bootstrap_resample",
]
