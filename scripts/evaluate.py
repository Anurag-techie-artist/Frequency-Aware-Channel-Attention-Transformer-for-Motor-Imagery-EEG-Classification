"""
Evaluation Script CLI Entry Point.
Phase 8: EEGMotorImageryModel Evaluation & Scientific Analysis.
"""

import os
import sys
import logging
import argparse

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.runner import EvaluationRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate EEGMotorImageryModel Checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/latest.pt",
        help="Path to trained model checkpoint file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to training config YAML file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluation",
        help="Directory to save evaluation reports and plots",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable generating visualization PNG plots",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Initializing evaluation run for checkpoint: {args.checkpoint}")

    runner = EvaluationRunner(
        checkpoint_path=args.checkpoint,
        train_config_path=args.config,
        output_dir=args.output_dir,
    )
    metrics, _ = runner.run(generate_plots=not args.no_plots)
    print(
        f"Evaluation execution finished. Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}"
    )


if __name__ == "__main__":
    main()
