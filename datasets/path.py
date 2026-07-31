"""
Centralized Dataset Path Resolution & Discovery Utility.
Phase 10 Patch v0.10.1: Eliminates hardcoded dataset paths across OS environments.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def get_project_root() -> str:
    """Find and return absolute path to project root directory."""
    curr = os.path.abspath(os.getcwd())
    while curr and not os.path.exists(os.path.join(curr, "models")):
        parent = os.path.dirname(curr)
        if parent == curr:
            break
        curr = parent
    return curr if os.path.exists(os.path.join(curr, "models")) else os.path.abspath(os.getcwd())


def resolve_dataset_path(path_str: str, project_root: Optional[str] = None) -> str:
    """
    Resolve dataset path string to an absolute path, expanding user ~ and environment variables.

    Args:
        path_str: Raw path string (relative or absolute)
        project_root: Optional project root path

    Returns:
        Resolved absolute path string
    """
    expanded = os.path.expanduser(os.path.expandvars(path_str))
    if os.path.isabs(expanded):
        return os.path.normpath(expanded)

    root = project_root if project_root else get_project_root()
    return os.path.normpath(os.path.join(root, expanded))


class DatasetPaths:
    """Centralized manager for HGD dataset path resolution and validation."""

    def __init__(self, config_path: Optional[str] = None, project_root: Optional[str] = None):
        self.project_root = project_root if project_root else get_project_root()
        self.config_path = config_path if config_path else os.path.join(self.project_root, "configs", "dataset.yaml")
        self.dataset_config = self._load_yaml_config()

    def _load_yaml_config(self) -> Dict[str, Any]:
        """Load configs/dataset.yaml safely if it exists."""
        if os.path.exists(self.config_path):
            try:
                import yaml
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    if isinstance(data, dict) and "dataset" in data:
                        return data["dataset"]
                    elif isinstance(data, dict):
                        return data
            except Exception as e:
                logger.warning(f"Could not load dataset config at {self.config_path}: {e}")
        return {}

    def get_dataset_root(self) -> str:
        """
        Resolve HGD dataset root directory following priority:
        1. Environment Variable: HGD_DATASET_ROOT
        2. configs/dataset.yaml (dataset.root)
        3. Fallback: ./hgd or <project_root>/hgd

        Returns:
            Resolved absolute dataset root directory path
        """
        # Priority 1: Environment Variable
        env_path = os.environ.get("HGD_DATASET_ROOT")
        if env_path and env_path.strip():
            resolved = resolve_dataset_path(env_path.strip(), self.project_root)
            logger.debug(f"Resolved HGD_DATASET_ROOT from env var: {resolved}")
            return resolved

        # Priority 2: configs/dataset.yaml
        yaml_root = self.dataset_config.get("root")
        if yaml_root and str(yaml_root).strip():
            resolved = resolve_dataset_path(str(yaml_root).strip(), self.project_root)
            logger.debug(f"Resolved HGD dataset root from configs/dataset.yaml: {resolved}")
            return resolved

        # Priority 3: Fallback ./hgd
        fallback = os.path.normpath(os.path.join(self.project_root, "hgd"))
        logger.debug(f"Resolved HGD dataset root from fallback: {fallback}")
        return fallback

    def get_train_directory(self) -> str:
        """Get relative or absolute train directory name."""
        return str(self.dataset_config.get("train_directory", "train1"))

    def get_test_directory(self) -> str:
        """Get relative or absolute test directory name."""
        return str(self.dataset_config.get("test_directory", "test1"))

    def validate_dataset(self, path: Optional[str] = None) -> bool:
        """
        Validate whether dataset root directory exists.

        Args:
            path: Optional dataset path to validate (defaults to get_dataset_root())

        Returns:
            True if valid, raises FileNotFoundError if missing
        """
        target = path if path else self.get_dataset_root()
        if not os.path.exists(target):
            msg = (
                f"\n[ERROR] High-Gamma Dataset (HGD) directory not found at: '{target}'\n"
                f"Please specify the dataset location using one of the following methods:\n"
                f"  Option 1 (Recommended): Set environment variable HGD_DATASET_ROOT=/path/to/hgd\n"
                f"  Option 2: Edit 'root' in configs/dataset.yaml\n"
                f"  Option 3: Place 'hgd/' folder inside repository root: '{self.project_root}'\n"
            )
            raise FileNotFoundError(msg)
        return True


# Convenience global resolution functions
def get_dataset_root(project_root: Optional[str] = None) -> str:
    """Global convenience function returning resolved absolute HGD dataset root path."""
    return DatasetPaths(project_root=project_root).get_dataset_root()


def get_train_directory(project_root: Optional[str] = None) -> str:
    """Global convenience function returning HGD train directory name."""
    return DatasetPaths(project_root=project_root).get_train_directory()


def get_test_directory(project_root: Optional[str] = None) -> str:
    """Global convenience function returning HGD test directory name."""
    return DatasetPaths(project_root=project_root).get_test_directory()


def validate_dataset(path: Optional[str] = None, project_root: Optional[str] = None) -> bool:
    """Global convenience function validating dataset root directory existence."""
    return DatasetPaths(project_root=project_root).validate_dataset(path=path)
