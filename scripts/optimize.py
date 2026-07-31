"""
Hyperparameter Optimization CLI Script Entry Point.
Phase 9: Extensible HPO Framework.
"""

import os
import sys
import logging
import argparse

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from hpo.runner import HPOExperimentRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Hyperparameter Optimization for EEGMotorImageryModel"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/hpo.yaml",
        help="Path to HPO config YAML file",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resuming from previous optimization run",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Initializing Hyperparameter Optimization run with config: {args.config}")

    runner = HPOExperimentRunner(hpo_config_path=args.config)
    scheduler = runner.run(resume=not args.no_resume)

    best_trial = scheduler.get_best_trial()
    if best_trial:
        print(f"\nHPO Run Complete! Best Trial #{best_trial.trial_id} | Score: {best_trial.score:.4f}")
        print(f"Best Parameters: {best_trial.params}")
    else:
        print("HPO Run Finished. No completed trials found.")


if __name__ == "__main__":
    main()
