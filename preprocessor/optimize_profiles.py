"""Optimize preprocessor lighting profiles against bbox fixtures."""

from __future__ import annotations

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR in sys.path:
    sys.path.remove(CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import argparse
import json
from pathlib import Path

from preprocessor._profile_optimization import (
    DEFAULT_MAX_SWEEPS,
    DEFAULT_SEED,
    evaluate_parameters,
    get_default_dataset_dir,
    get_default_parameters,
    load_dataset,
    optimize_parameters,
    parameters_to_json_dict,
    runtime_thresholds_from_cutoff,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Run local-search optimization for normal and low-light "
        "skin-fusion profiles against the sample-frame bbox fixtures.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=str(get_default_dataset_dir()),
        help="Directory containing sample-frame images and estimated_bbox.json.",
    )
    parser.add_argument(
        "--max-sweeps",
        type=int,
        default=DEFAULT_MAX_SWEEPS,
        help="Maximum number of hill-climb sweeps to run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Deterministic seed used to break tie order during search.",
    )
    return parser


def main() -> int:
    """Run the optimization CLI."""
    parser = build_parser()
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    samples = load_dataset(dataset_dir)
    starting_parameters = get_default_parameters()
    outcome = optimize_parameters(
        samples,
        starting_parameters=starting_parameters,
        max_sweeps=args.max_sweeps,
        seed=args.seed,
    )

    final_report = evaluate_parameters(
        outcome.parameters,
        samples,
        dataset_dir=dataset_dir,
        include_per_image=False,
    )
    runtime_enter, runtime_exit = runtime_thresholds_from_cutoff(
        outcome.parameters.brightness_cutoff
    )
    console_report = dict(final_report)
    console_report.update(
        {
            "starting_parameters": parameters_to_json_dict(starting_parameters),
            "starting_dataset_score": outcome.starting_score,
            "seed": args.seed,
            "max_sweeps": args.max_sweeps,
            "sweeps_completed": outcome.sweeps_completed,
            "final_step_sizes": outcome.step_sizes,
            "improved": outcome.improved,
            "rewrote_source": False,
            "runtime_thresholds": {
                "enter_low_light_threshold": runtime_enter,
                "exit_low_light_threshold": runtime_exit,
            },
        }
    )
    print(json.dumps(console_report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
