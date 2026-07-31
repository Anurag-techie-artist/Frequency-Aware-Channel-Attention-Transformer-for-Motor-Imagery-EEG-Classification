"""
Unit Tests for AugmentationExperimentRunner (Phase 10).
"""

import os
import sys
import unittest
import tempfile

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from augmentation.runner import AugmentationExperimentRunner


class TestAugmentationRunner(unittest.TestCase):
    """Test suite for AugmentationExperimentRunner execution."""

    def test_augmentation_runner_execution(self):
        """Test executing AugmentationExperimentRunner with synthetic setup."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            clean_out_dir = tmp_dir.replace("\\", "/")
            config_path = os.path.join(tmp_dir, "test_gan.yaml")
            yaml_content = f"""
augmentation:
  strategy: wgan_gp
  ratio: 0.25
  seed: 42

gan:
  latent_dim: 16
  num_classes: 4
  generator_hidden_dim: 32
  critic_hidden_dim: 32
  critic_steps: 1
  epochs: 1
  batch_size: 8
  learning_rate: 0.001
  device: cpu

output:
  output_dir: {clean_out_dir}

model:
  num_channels: 5
  num_bands: 2
  num_samples: 20
  transformer:
    num_layers: 1
    d_model: 32
  classifier:
    hidden_dim: 32

training:
  synthetic_data: true
  batch_size: 8
  epochs: 1
  mixed_precision: false
  device: cpu
"""
            with open(config_path, "w", encoding="utf-8") as f:
                f.write(yaml_content)

            runner = AugmentationExperimentRunner(gan_config_path=config_path)
            report = runner.run()

            self.assertIn("psd_similarity", report)
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "synthetic", "generated_dataset.pt")))


if __name__ == "__main__":
    unittest.main()
