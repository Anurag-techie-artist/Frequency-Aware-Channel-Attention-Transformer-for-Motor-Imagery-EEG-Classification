"""
Evaluation Framework Package.
"""

from evaluation.inference import InferenceEngine, InferenceResults
from evaluation.manifest import generate_manifest
from evaluation.embeddings import EmbeddingProjector
from evaluation.statistics import compute_summary_statistics
from evaluation.report import ReportGenerator
from evaluation.evaluator import Evaluator
from evaluation.runner import EvaluationRunner

__all__ = [
    "InferenceEngine",
    "InferenceResults",
    "generate_manifest",
    "EmbeddingProjector",
    "compute_summary_statistics",
    "ReportGenerator",
    "Evaluator",
    "EvaluationRunner",
]
