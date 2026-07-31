"""
Unit Tests for Centralized Dataset Path Resolution & Discovery (Phase 10 Patch v0.10.1).
"""

import os
import sys
import tempfile
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.path import (
    DatasetPaths,
    get_dataset_root,
    resolve_dataset_path,
    validate_dataset,
)
from configs.config_loader import load_dataset_config, load_master_config


class TestDatasetPaths(unittest.TestCase):
    """Test suite for dataset path discovery, precedence hierarchy, and validation."""

    def setUp(self):
        # Save original environment variable if present
        self.orig_env = os.environ.get("HGD_DATASET_ROOT")

    def tearDown(self):
        # Restore original environment variable
        if self.orig_env is not None:
            os.environ["HGD_DATASET_ROOT"] = self.orig_env
        elif "HGD_DATASET_ROOT" in os.environ:
            del os.environ["HGD_DATASET_ROOT"]

    def test_env_var_precedence(self):
        """Test HGD_DATASET_ROOT environment variable takes highest priority."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            os.environ["HGD_DATASET_ROOT"] = tmp_dir
            resolved = get_dataset_root(project_root=PROJECT_ROOT)
            self.assertEqual(os.path.normpath(resolved), os.path.normpath(tmp_dir))

    def test_yaml_config_loading(self):
        """Test dataset configuration YAML parsing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yaml_path = os.path.join(tmp_dir, "dataset.yaml")
            with open(yaml_path, "w", encoding="utf-8") as f:
                f.write("dataset:\n  root: './custom_hgd'\n  train_directory: 'train1'\n")

            if "HGD_DATASET_ROOT" in os.environ:
                del os.environ["HGD_DATASET_ROOT"]

            paths = DatasetPaths(config_path=yaml_path, project_root=PROJECT_ROOT)
            root = paths.get_dataset_root()
            expected = os.path.normpath(os.path.join(PROJECT_ROOT, "custom_hgd"))
            self.assertEqual(os.path.normpath(root), expected)

    def test_fallback_to_hgd(self):
        """Test falling back to ./hgd when no env var or YAML root is provided."""
        if "HGD_DATASET_ROOT" in os.environ:
            del os.environ["HGD_DATASET_ROOT"]

        paths = DatasetPaths(config_path="/nonexistent/dataset.yaml", project_root=PROJECT_ROOT)
        root = paths.get_dataset_root()
        expected = os.path.normpath(os.path.join(PROJECT_ROOT, "hgd"))
        self.assertEqual(os.path.normpath(root), expected)

    def test_validation_error_on_missing(self):
        """Test validate_dataset raises FileNotFoundError with clear error message when missing."""
        missing_path = os.path.join(PROJECT_ROOT, "non_existent_hgd_dir_12345")
        with self.assertRaises(FileNotFoundError) as ctx:
            validate_dataset(path=missing_path, project_root=PROJECT_ROOT)

        err_msg = str(ctx.exception)
        self.assertIn("HGD_DATASET_ROOT", err_msg)
        self.assertIn("configs/dataset.yaml", err_msg)

    def test_absolute_path_resolution(self):
        """Test resolve_dataset_path handles relative, absolute, and user paths."""
        abs_p = resolve_dataset_path("/tmp/test_dir", project_root=PROJECT_ROOT)
        self.assertTrue(os.path.isabs(abs_p))

        rel_p = resolve_dataset_path("hgd", project_root=PROJECT_ROOT)
        self.assertTrue(os.path.isabs(rel_p))
        self.assertEqual(os.path.normpath(rel_p), os.path.normpath(os.path.join(PROJECT_ROOT, "hgd")))

    def test_master_config_includes_dataset(self):
        """Test load_master_config merges dataset settings."""
        master = load_master_config(project_root=PROJECT_ROOT)
        self.assertIn("dataset", master)
        self.assertEqual(master["dataset"].get("train_directory"), "train1")


if __name__ == "__main__":
    unittest.main()
