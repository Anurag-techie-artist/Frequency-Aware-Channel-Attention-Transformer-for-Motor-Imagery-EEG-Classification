"""
TensorBoard SummaryWriter Logger Backend.
"""

import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class TensorBoardLogger:
    """Wrapper around PyTorch SummaryWriter for TensorBoard logging."""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.writer = None

        try:
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
        except ImportError:
            logger.warning(
                "tensorboard not installed. TensorBoard logging will be disabled."
            )

    def log_scalars(self, metrics: Dict[str, float], step: int):
        """Log scalar metrics at step/epoch."""
        if self.writer is None:
            return
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(k, v, step)

    def close(self):
        """Close writer connection."""
        if self.writer:
            self.writer.close()
