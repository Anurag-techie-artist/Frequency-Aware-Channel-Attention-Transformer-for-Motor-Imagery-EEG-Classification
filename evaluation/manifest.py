"""
Experiment Manifest Generator.

Generates manifest.json tracking git commit hash, model version, checkpoint name,
config hash, evaluation timestamp, PyTorch version, and CUDA device status.
"""

import os
import json
import time
import hashlib
import subprocess
from typing import Dict, Any, Optional

import torch


def get_git_commit_hash() -> str:
    """Extract current git commit hash."""
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("utf-8").strip()
        return output
    except Exception:
        return "unknown_commit"


def compute_config_hash(config: Dict[str, Any]) -> str:
    """Compute MD5 hash of configuration dictionary."""
    cfg_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(cfg_str.encode("utf-8")).hexdigest()


def generate_manifest(
    checkpoint_name: str = "best.pt",
    config: Optional[Dict[str, Any]] = None,
    model_version: str = "v0.8.0",
    output_dir: str = "outputs/evaluation",
) -> Dict[str, Any]:
    """
    Generate manifest dictionary and save manifest.json to output_dir.

    Args:
        checkpoint_name: Name of evaluated checkpoint
        config: Configuration dictionary
        model_version: Framework version string
        output_dir: Output directory path

    Returns:
        Manifest metadata dictionary
    """
    config_dict = config or {}
    manifest = {
        "git_commit": get_git_commit_hash(),
        "model_version": model_version,
        "checkpoint": checkpoint_name,
        "config_hash": compute_config_hash(config_dict),
        "evaluation_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest
