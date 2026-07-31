"""
Unit Tests for HyperparameterOptimizer (Phase 9).
"""

import os
import sys
import tempfile
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from hpo.optimizer import HyperparameterOptimizer


class TestHPOOptimizer(unittest.TestCase):
    """Test suite for HyperparameterOptimizer execution and outputs."""

    def test_hpo_optimization_run(self):
        """Test executing a 2-trial HPO run creates summary, best_config.yaml, and leaderboard."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = {
                "hpo": {
                    "seed": 42,
                    "strategy": "random",
                    "metric": "val_accuracy",
                    "mode": "max",
                    "n_trials": 2,
                    "max_epochs_per_trial": 1,
                    "output_dir": tmp_dir,
                },
                "search_space": {
                    "d_model": {"type": "categorical", "values": [32, 64]},
                },
                "model": {
                    "num_channels": 10,
                    "num_bands": 2,
                    "transformer": {"num_layers": 1, "d_model": 32},
                    "classifier": {"hidden_dim": 32},
                },
                "training": {
                    "synthetic_data": True,
                    "batch_size": 8,
                    "mixed_precision": False,
                    "device": "cpu",
                },
            }

            optimizer = HyperparameterOptimizer(config)
            sched = optimizer.optimize(resume=False)

            self.assertEqual(len(sched.trials), 2)
            best_trial = sched.get_best_trial()
            self.assertIsNotNone(best_trial)

            # Check output files exist
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "leaderboard.csv")))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "trials.csv")))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "best_config.yaml")))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "summary.json")))


if __name__ == "__main__":
    unittest.main()
