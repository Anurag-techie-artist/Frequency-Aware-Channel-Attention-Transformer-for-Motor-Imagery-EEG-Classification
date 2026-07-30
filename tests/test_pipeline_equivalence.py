"""
Unit Tests for Verifying Pipeline Equivalence Against Baseline (Phase 2).

Compares outputs of EEGPreprocessingPipeline against baseline load_edf and create_windows
functions from basefile.py to ensure 100% numerical and functional equivalence.
"""

import os
import sys
import unittest
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.pipeline import EEGPreprocessingPipeline
from datasets.loader import load_raw_edf, extract_events
from datasets.preprocessing import resample_signal, bandpass_filter, extract_epochs
from datasets.windowing import generate_sliding_windows


# Baseline reference implementation functions matching basefile.py exactly
def baseline_load_edf(file_path: str):
    import mne
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    raw.resample(250)
    raw.filter(4, 38, fir_design="firwin", verbose=False)

    events, event_dict = mne.events_from_annotations(raw, verbose=False)
    event_keys = sorted(list(event_dict.keys()))
    label_map = {event_dict[key]: idx for idx, key in enumerate(event_keys)}

    epochs = mne.Epochs(
        raw, events, event_id=event_dict, tmin=0.5, tmax=3.5,
        baseline=None, preload=True, verbose=False
    )
    X = epochs.get_data()
    y_raw = epochs.events[:, -1]
    y = np.array([label_map[x] for x in y_raw])
    return X, y


def baseline_create_windows(X: np.ndarray, y: np.ndarray):
    X_out = []
    y_out = []
    trial_ids = []

    for trial_idx in range(len(X)):
        trial = X[trial_idx].copy()
        mean = np.mean(trial, axis=1, keepdims=True)
        std = np.std(trial, axis=1, keepdims=True)
        trial = (trial - mean) / (std + 1e-6)

        for ws in range(0, trial.shape[1] - 250, 50):
            we = ws + 250
            window = trial[:, ws:we]
            X_out.append(window)
            y_out.append(y[trial_idx])
            trial_ids.append(trial_idx)

    return np.array(X_out), np.array(y_out), np.array(trial_ids)


class TestPipelineEquivalence(unittest.TestCase):
    """Test suite comparing refactored EEGPreprocessingPipeline against baseline."""

    def setUp(self):
        self.sample_edf = os.path.join(PROJECT_ROOT, "hgd", "train1", "1.edf")
        self.assertTrue(
            os.path.exists(self.sample_edf),
            f"Sample EDF file not found at {self.sample_edf}"
        )

    def test_pipeline_equivalence(self):
        """Verify refactored pipeline produces 100% equivalent outputs to baseline."""
        # 1. Run baseline
        X_base_epochs, y_base_epochs = baseline_load_edf(self.sample_edf)
        X_base_win, y_base_win, base_trial_ids = baseline_create_windows(X_base_epochs, y_base_epochs)

        # 2. Run EEGPreprocessingPipeline
        pipeline = EEGPreprocessingPipeline()
        X_mod_win, y_mod_win, mod_trial_ids = pipeline.process(self.sample_edf)

        # 3. Assertions
        # Epoch count equivalence
        self.assertEqual(len(X_base_epochs), len(y_base_epochs))
        
        # Window shape equivalence
        self.assertEqual(X_base_win.shape, X_mod_win.shape, "Window shapes do not match!")
        self.assertEqual(y_base_win.shape, y_mod_win.shape, "Window label shapes do not match!")
        self.assertEqual(base_trial_ids.shape, mod_trial_ids.shape, "Trial ID shapes do not match!")

        # Label equivalence
        np.testing.assert_array_equal(
            y_base_win, y_mod_win,
            err_msg="Modular window labels do not match baseline!"
        )

        # Trial ID equivalence
        np.testing.assert_array_equal(
            base_trial_ids, mod_trial_ids,
            err_msg="Modular trial IDs do not match baseline!"
        )

        # Numerical signal values equivalence
        np.testing.assert_allclose(
            X_base_win, X_mod_win, rtol=1e-5, atol=1e-5,
            err_msg="Modular window signal data values do not match baseline!"
        )
        print("\n[PASSED] Pipeline equivalence test: 100% match with baseline!")


if __name__ == "__main__":
    unittest.main()
