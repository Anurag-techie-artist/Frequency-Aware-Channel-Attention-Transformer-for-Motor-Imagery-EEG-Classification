"""
Centralized Configuration Loader and Merger.

Loads and merges YAML configuration files (preprocessing.yaml, model.yaml, train.yaml).
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def load_yaml_config(path: str) -> Dict[str, Any]:
    """Load dictionary from YAML file path."""
    if not os.path.exists(path):
        logger.warning(f"Config file {path} not found. Returning empty dict.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge multiple dictionaries. Later dictionaries override earlier ones.
    """
    merged: Dict[str, Any] = {}
    for cfg in configs:
        for k, v in cfg.items():
            if isinstance(v, dict) and k in merged and isinstance(merged[k], dict):
                merged[k] = merge_configs(merged[k], v)
            else:
                merged[k] = v
    return merged


def load_master_config(
    train_cfg_path: Optional[str] = None,
    model_cfg_path: Optional[str] = None,
    prep_cfg_path: Optional[str] = None,
    project_root: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load master merged configuration combining preprocessing, model, and training YAMLs.
    """
    if project_root is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    if prep_cfg_path is None:
        prep_cfg_path = os.path.join(project_root, "configs", "preprocessing.yaml")
    if model_cfg_path is None:
        model_cfg_path = os.path.join(project_root, "configs", "model.yaml")
    if train_cfg_path is None:
        train_cfg_path = os.path.join(project_root, "configs", "train.yaml")

    prep_cfg = load_yaml_config(prep_cfg_path)
    model_cfg = load_yaml_config(model_cfg_path)
    train_cfg = load_yaml_config(train_cfg_path)

    master = merge_configs(prep_cfg, model_cfg, train_cfg)
    return master
