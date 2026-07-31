"""
GANState Dataclass for Tracking Training Progress and Checkpoint Recovery.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import torch


@dataclass
class GANState:
    """Encapsulates execution state for Conditional WGAN-GP training."""

    epoch: int
    global_step: int
    generator_state: Dict[str, Any]
    critic_state: Dict[str, Any]
    g_optimizer_state: Dict[str, Any]
    c_optimizer_state: Dict[str, Any]
    best_loss: float = float("inf")
    history: List[Dict[str, Any]] = field(default_factory=list)
