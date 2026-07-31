"""
Unit Tests for HPO Optimization Resumption (Phase 9).
"""

import os
import sys
import tempfile
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from hpo.optimizer import HyperparameterOptimizer


class TestHPOResume(unittest.TestCase):
    """Test suite for resuming interrupted HPO optimization runs."""

    def test_hpo_resumption(self):
        """Test running 1 trial, then resuming to run total 2 trials advances trial_id."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = {
                "hpo": {
                    "seed": 42,
                    "strategy": "random",
                    "metric": "val_accuracy",
                    "mode": "max",
                    "n_trials": 1,
                    "max_epochs_per_trial": 1,
                    "output_dir": tmp_dir,
                },
                "search_space": {
                    "d_model": {"type": "categorical", "values": [32]},
                },
                "model": {
                    "num_channels": 10,
                    "num_bands": 2,
                    "transformer": {"num_layers": 1, "d_model": 32},
                    "classifier": {"hidden_dim": 32},
                },
                "training": {
                    "batch_size": 8,
                    "mixed_precision": False,
                    "device": "cpu",
                },
            }

            opt1 = HyperparameterOptimizer(config)
            sched1 = opt1.optimize(resume=False)
            self.assertEqual(len(sched1.trials), 1)

            # Resume with n_trials = 2
            config["hpo"]["n_trials"] = 2
            opt2 = HyperparameterOptimizer(config)
            sched2 = opt2.optimize(resume=True)

            self.assertEqual(len(sched2.trials), 2)
            self.assertEqual(sched2.trials[0].trial_id, 0)
            self.assertEqual(sched2.trials[1].trial_id, 1)


if __name__ == "__main__":
    unittest.main()
