"""
Unified ExperimentLogger orchestrating CSV and TensorBoard logging backends.
"""

import os
from typing import Dict, Any, Optional
from loggers.csv_logger import CSVLogger
from loggers.tensorboard_logger import TensorBoardLogger


class ExperimentLogger:
    """Orchestrates multi-backend metric logging (Console, CSV, TensorBoard)."""

    def __init__(
        self,
        log_dir: str = "outputs/logs",
        use_csv: bool = True,
        use_tensorboard: bool = True,
    ):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.csv_logger = (
            CSVLogger(os.path.join(log_dir, "metrics.csv")) if use_csv else None
        )
        self.tb_logger = (
            TensorBoardLogger(os.path.join(log_dir, "tb")) if use_tensorboard else None
        )

    def log_metrics(self, metrics: Dict[str, Any], epoch: int):
        """Log epoch metrics across enabled backends."""
        if self.csv_logger:
            self.csv_logger.log_epoch(metrics)
        if self.tb_logger:
            self.tb_logger.log_scalars(metrics, step=epoch)

    def close(self):
        """Close loggers."""
        if self.tb_logger:
            self.tb_logger.close()
