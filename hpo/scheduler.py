"""
TrialScheduler Module for Execution State Management & Resumption.
"""

from typing import Dict, Any, List, Optional
from hpo.trial import Trial, TrialStatus


class TrialScheduler:
    """Manages trial queue ordering, state tracking, and optimization resumption."""

    def __init__(self, max_trials: int = 20):
        self.max_trials = max_trials
        self.trials: List[Trial] = []

    def add_trial(self, trial_id: int, params: Dict[str, Any]) -> Trial:
        """Create and queue a new Trial in PENDING status."""
        trial = Trial(trial_id=trial_id, params=params, status=TrialStatus.PENDING)
        self.trials.append(trial)
        return trial

    def get_trial(self, trial_id: int) -> Optional[Trial]:
        """Get trial by trial_id."""
        for t in self.trials:
            if t.trial_id == trial_id:
                return t
        return None

    def mark_running(self, trial_id: int) -> Trial:
        """Update trial status to RUNNING."""
        trial = self.get_trial(trial_id)
        if trial is None:
            raise KeyError(f"Trial ID {trial_id} not found in scheduler.")
        updated = trial.start()
        self.trials[trial_id] = updated
        return updated

    def mark_completed(
        self,
        trial_id: int,
        score: float,
        metrics: Dict[str, Any],
        checkpoint_path: Optional[str] = None,
        duration_seconds: float = 0.0,
    ) -> Trial:
        """Update trial status to COMPLETED."""
        trial = self.get_trial(trial_id)
        if trial is None:
            raise KeyError(f"Trial ID {trial_id} not found in scheduler.")
        updated = trial.complete(
            score=score,
            metrics=metrics,
            checkpoint_path=checkpoint_path,
            duration_seconds=duration_seconds,
        )
        self.trials[trial_id] = updated
        return updated

    def mark_failed(
        self, trial_id: int, error_message: str, duration_seconds: float = 0.0
    ) -> Trial:
        """Update trial status to FAILED."""
        trial = self.get_trial(trial_id)
        if trial is None:
            raise KeyError(f"Trial ID {trial_id} not found in scheduler.")
        updated = trial.fail(error_message=error_message, duration_seconds=duration_seconds)
        self.trials[trial_id] = updated
        return updated

    def get_completed_trials(self) -> List[Trial]:
        """Return list of completed trials."""
        return [t for t in self.trials if t.status == TrialStatus.COMPLETED]

    def get_best_trial(self, mode: str = "max") -> Optional[Trial]:
        """Return trial with best score."""
        completed = self.get_completed_trials()
        if not completed:
            return None
        if mode == "max":
            return max(completed, key=lambda t: t.score if t.score is not None else -float("inf"))
        return min(completed, key=lambda t: t.score if t.score is not None else float("inf"))
