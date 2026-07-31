"""
GAN Training & EEG Data Augmentation CLI Script Entry Point.
Phase 10: EEG Data Augmentation Framework (v0.10.0 RC1).
"""

import os
import sys
import logging
import argparse

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from augmentation.runner import AugmentationExperimentRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run WGAN-GP Training & EEG Data Augmentation Framework"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/gan.yaml",
        help="Path to GAN config YAML file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Initializing EEG Data Augmentation Experiment with config: {args.config}")

    runner = AugmentationExperimentRunner(gan_config_path=args.config)
    eval_report = runner.run()

    print("\n=== Augmentation Experiment Finished ===")
    for k, v in eval_report.items():
        print(f"  - {k:<25}: {v:.4f}")


if __name__ == "__main__":
    main()
