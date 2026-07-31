"""
Training Script CLI Entry Point.
Phase 7: EEGMotorImageryModel Training Framework.
"""

import os
import sys
import logging
import argparse

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.runner import ExperimentRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train EEGMotorImageryModel Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to training config YAML file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Initializing training run with config: {args.config}")

    runner = ExperimentRunner(train_config_path=args.config)
    runner.run(resume_path=args.resume)
    print("Training script execution finished.")


if __name__ == "__main__":
    main()
