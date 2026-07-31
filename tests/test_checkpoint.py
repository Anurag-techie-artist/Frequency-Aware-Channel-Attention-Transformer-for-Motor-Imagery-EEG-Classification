"""
Unit Tests for CheckpointManager (Phase 7).
"""

import os
import sys
import tempfile
import unittest
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.checkpoint import CheckpointManager
from training.state import TrainState


class TestCheckpoint(unittest.TestCase):
    """Test suite for CheckpointManager."""

    def test_checkpoint_save_and_load(self):
        """Test full state dictionary saving and restoration."""
        model = nn.Linear(10, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        state = TrainState(epoch=5, global_step=50, best_metric=0.85)

        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = CheckpointManager(save_dir=tmp_dir)

            filepath = manager.save_checkpoint(
                model=model, train_state=state, optimizer=optimizer
            )
            self.assertTrue(os.path.exists(filepath))

            # Reconstruct into new instances
            new_model = nn.Linear(10, 2)
            new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
            loaded_state = manager.load_checkpoint(
                checkpoint_path=filepath, model=new_model, optimizer=new_optimizer
            )

            self.assertEqual(loaded_state.epoch, 5)
            self.assertEqual(loaded_state.global_step, 50)
            self.assertEqual(loaded_state.best_metric, 0.85)

            # Check model weights match
            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                self.assertTrue(torch.equal(p1, p2))


if __name__ == "__main__":
    unittest.main()
