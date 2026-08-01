"""
Regression Tests for Trial-Level Train/Validation Split (v0.11.2).

Verifies zero train-validation leakage at the trial level:
- Disjoint trial sets (train_trials ∩ val_trials == ∅)
- All validation windows originate strictly from validation trials
- All training windows originate strictly from training trials
- Conservation of trial count (len(train) + len(val) == len(total))
- No duplicate trial IDs
- Deterministic reproducible splitting given identical seed
"""

import os
import sys
import unittest
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.builder import build_dataloaders
from configs.config_loader import load_master_config


class TestTrialSplit(unittest.TestCase):
    """Test suite verifying trial-level validation split integrity and zero data leakage."""

    @classmethod
    def setUpClass(cls):
        cls.config = load_master_config(project_root=PROJECT_ROOT)

    def test_trial_split_disjointness_and_conservation(self):
        """Tests 1, 4, 5: Verify trial sets are disjoint, non-duplicated, and total trial count conserved."""
        train_loader, val_loader, _ = build_dataloaders(self.config, project_root=PROJECT_ROOT)
        train_ds = train_loader.dataset
        val_ds = val_loader.dataset

        train_trials = set(train_ds.included_trials)
        val_trials = set(val_ds.included_trials)

        # Test 1: Intersection must be empty
        intersection = train_trials.intersection(val_trials)
        self.assertEqual(
            len(intersection),
            0,
            f"Train and validation trial sets overlap! Shared trials: {intersection}",
        )

        # Test 5: No duplicate trial IDs in individual lists
        self.assertEqual(len(train_trials), len(train_ds.included_trials), "Duplicate trial IDs found in train_ds")
        self.assertEqual(len(val_trials), len(val_ds.included_trials), "Duplicate trial IDs found in val_ds")

        # Test 4: Total trial count conservation
        total_trials = len(train_trials) + len(val_trials)
        # Check against metadata
        all_entries = train_ds.metadata.get("files", [])
        requested_abs = set(os.path.abspath(p) for p in train_ds.file_paths)
        expected_total = sum(
            e["num_trials"] for e in all_entries if os.path.abspath(e["edf_path"]) in requested_abs
        )
        self.assertEqual(total_trials, expected_total, f"Total trial count mismatch: {total_trials} vs {expected_total}")

    def test_window_origin_integrity(self):
        """Tests 2 & 3: Verify training and validation windows originate strictly from their respective trial sets."""
        train_loader, val_loader, _ = build_dataloaders(self.config, project_root=PROJECT_ROOT)
        train_ds = train_loader.dataset
        val_ds = val_loader.dataset

        train_trial_set = set(train_ds.included_trials)
        val_trial_set = set(val_ds.included_trials)

        # Test 3: Every training window originates only from training trials
        for sample in train_ds._window_samples:
            cache_path, trial_idx, start_sample, abs_edf = sample
            trial_key = (abs_edf, trial_idx)
            self.assertIn(
                trial_key,
                train_trial_set,
                f"Training window sample originated from non-training trial: {trial_key}",
            )
            self.assertNotIn(
                trial_key,
                val_trial_set,
                f"Training window sample originated from validation trial: {trial_key}",
            )

        # Test 2: Every validation window originates only from validation trials
        for sample in val_ds._window_samples:
            cache_path, trial_idx, start_sample, abs_edf = sample
            trial_key = (abs_edf, trial_idx)
            self.assertIn(
                trial_key,
                val_trial_set,
                f"Validation window sample originated from non-validation trial: {trial_key}",
            )
            self.assertNotIn(
                trial_key,
                train_trial_set,
                f"Validation window sample originated from training trial: {trial_key}",
            )

    def test_deterministic_split(self):
        """Test 6: Deterministic trial split given identical seed=42."""
        tr1, val1, _ = build_dataloaders(self.config, project_root=PROJECT_ROOT)
        tr2, val2, _ = build_dataloaders(self.config, project_root=PROJECT_ROOT)

        self.assertEqual(tr1.dataset.included_trials, tr2.dataset.included_trials)
        self.assertEqual(val1.dataset.included_trials, val2.dataset.included_trials)


if __name__ == "__main__":
    unittest.main()
