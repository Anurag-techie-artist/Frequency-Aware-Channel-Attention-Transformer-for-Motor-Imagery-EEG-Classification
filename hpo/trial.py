"""
TrialStatus Enum and Immutable Trial Dataclass.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


class TrialStatus(Enum):
    """Execution status for an HPO trial."""

    PENDING = 0
    RUNNING = 1
    COMPLETED = 2
    FAILED = 3
    PRUNED = 4


@dataclass(frozen=True)
class Trial:
    """Immutable representation of a hyperparameter optimization trial."""

    trial_id: int
    params: Dict[str, Any]
    status: TrialStatus = TrialStatus.PENDING
    score: Optional[float] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    checkpoint_path: Optional[str] = None
    duration_seconds: float = 0.0
    error_message: Optional[str] = None

    def start(self) -> "Trial":
        """Return a new Trial instance with RUNNING status."""
        return Trial(
            trial_id=self.trial_id,
            params=self.params,
            status=TrialStatus.RUNNING,
            score=self.score,
            metrics=self.metrics,
            checkpoint_path=self.checkpoint_path,
            duration_seconds=self.duration_seconds,
            error_message=self.error_message,
        )

    def complete(
        self,
        score: float,
        metrics: Dict[str, Any],
        checkpoint_path: Optional[str] = None,
        duration_seconds: float = 0.0,
    ) -> "Trial":
        """Return a new Trial instance with COMPLETED status and recorded metrics."""
        return Trial(
            trial_id=self.trial_id,
            params=self.params,
            status=TrialStatus.COMPLETED,
            score=score,
            metrics=metrics,
            checkpoint_path=checkpoint_path,
            duration_seconds=duration_seconds,
        )

    def fail(self, error_message: str, duration_seconds: float = 0.0) -> "Trial":
        """Return a new Trial instance with FAILED status."""
        return Trial(
            trial_id=self.trial_id,
            params=self.params,
            status=TrialStatus.FAILED,
            score=None,
            metrics={},
            duration_seconds=duration_seconds,
            error_message=error_message,
        )
