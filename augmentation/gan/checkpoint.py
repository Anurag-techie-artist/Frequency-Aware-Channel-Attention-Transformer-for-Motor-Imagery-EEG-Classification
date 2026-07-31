"""
GANCheckpointManager for Saving and Restoring Generator and Critic Models.
"""

import os
import torch
from typing import Dict, Any, Optional
from augmentation.gan.state import GANState


class GANCheckpointManager:
    """Manages saving and loading of Generator and Critic PyTorch checkpoints."""

    def __init__(self, save_dir: str = "outputs/augmentation/gan"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def save_checkpoint(
        self,
        generator: torch.nn.Module,
        critic: torch.nn.Module,
        g_optimizer: torch.optim.Optimizer,
        c_optimizer: torch.optim.Optimizer,
        epoch: int,
        global_step: int,
        history: list,
        filename: str = "generator.pt",
    ) -> str:
        """Save generator and critic state dictionaries to checkpoint file."""
        state_dict = {
            "epoch": epoch,
            "global_step": global_step,
            "generator_state": generator.state_dict(),
            "critic_state": critic.state_dict(),
            "g_optimizer_state": g_optimizer.state_dict(),
            "c_optimizer_state": c_optimizer.state_dict(),
            "history": history,
        }
        ckpt_path = os.path.join(self.save_dir, filename)
        torch.save(state_dict, ckpt_path)

        # Also save separate generator.pt and critic.pt weights
        g_path = os.path.join(self.save_dir, "generator.pt")
        c_path = os.path.join(self.save_dir, "critic.pt")
        torch.save(generator.state_dict(), g_path)
        torch.save(critic.state_dict(), c_path)

        return ckpt_path

    def load_checkpoint(
        self,
        generator: torch.nn.Module,
        critic: torch.nn.Module,
        g_optimizer: Optional[torch.optim.Optimizer] = None,
        c_optimizer: Optional[torch.optim.Optimizer] = None,
        ckpt_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load checkpoint state into generator and critic."""
        path = ckpt_path if ckpt_path else os.path.join(self.save_dir, "generator.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"GAN checkpoint not found at {path}")

        checkpoint = torch.load(path, map_location="cpu")
        if "generator_state" in checkpoint:
            generator.load_state_dict(checkpoint["generator_state"])
            critic.load_state_dict(checkpoint["critic_state"])
            if g_optimizer and "g_optimizer_state" in checkpoint:
                g_optimizer.load_state_dict(checkpoint["g_optimizer_state"])
            if c_optimizer and "c_optimizer_state" in checkpoint:
                c_optimizer.load_state_dict(checkpoint["c_optimizer_state"])
        else:
            generator.load_state_dict(checkpoint)

        return checkpoint
