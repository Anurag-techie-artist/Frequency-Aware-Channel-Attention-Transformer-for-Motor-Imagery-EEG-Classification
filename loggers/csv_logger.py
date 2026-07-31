"""
CSV Metric Logger.
"""

import os
import csv
from typing import Dict, Any


class CSVLogger:
    """Logs epoch metrics to a CSV file."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.fieldnames = None

    def log_epoch(self, metrics: Dict[str, Any]):
        """Append epoch metrics row to CSV file."""
        if not self.fieldnames:
            self.fieldnames = list(metrics.keys())
            file_exists = os.path.exists(self.filepath)
            with open(self.filepath, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                if not file_exists:
                    writer.writeheader()

        with open(self.filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(metrics)
