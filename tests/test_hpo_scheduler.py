"""
Unit Tests for TrialScheduler (Phase 9).
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from hpo.scheduler import TrialScheduler
from hpo.trial import TrialStatus


class TestHPOScheduler(unittest.TestCase):
    """Test suite for TrialScheduler queue and status transitions."""

    def test_trial_scheduler_lifecycle(self):
        """Test scheduler creates, starts, and completes trials."""
        sched = TrialScheduler(max_trials=5)
        t0 = sched.add_trial(0, {"lr": 1e-3})
        self.assertEqual(t0.status, TrialStatus.PENDING)

        t0_run = sched.mark_running(0)
        self.assertEqual(t0_run.status, TrialStatus.RUNNING)

        t0_done = sched.mark_completed(0, score=0.88, metrics={"accuracy": 0.88})
        self.assertEqual(t0_done.status, TrialStatus.COMPLETED)
        self.assertEqual(t0_done.score, 0.88)

        best = sched.get_best_trial()
        self.assertIsNotNone(best)
        self.assertEqual(best.trial_id, 0)


if __name__ == "__main__":
    unittest.main()
