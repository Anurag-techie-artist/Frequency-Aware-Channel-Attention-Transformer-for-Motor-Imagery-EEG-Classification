"""
ObjectiveResult Container Dataclass.

Encapsulates optimization target score, full metric dictionary, checkpoint path,
execution duration, and epoch metric history.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass(frozen=True)
class ObjectiveResult:
    """Immutable container for evaluation objective results from a single trial."""

    score: float
    metrics: Dict[str, Any]
    checkpoint_path: Optional[str] = None
    duration_seconds: float = 0.0
    history: List[Dict[str, Any]] = field(default_factory=list)
