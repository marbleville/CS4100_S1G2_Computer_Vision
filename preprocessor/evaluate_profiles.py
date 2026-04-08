"""Score preprocessor profiles against the sample-frame bounding-box fixtures."""

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
from pathlib import Path

from preprocessor._profile_optimization import (
    evaluate_parameters,
    get_default_dataset_dir,
    get_default_parameters,
    load_dataset,
    load_parameters_from_json,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate normal and low-light skin-fusion profiles "
        "against the sample-frame bbox fixtures.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=str(get_default_dataset_dir()),
        help="Directory containing sample-frame images and estimated_bbox.json.",
    )
    parser.add_argument(
        "--params-json",
        help="Optional JSON file containing `normal_skin_profile`, "
        "`low_light_skin_profile`, and `brightness_cutoff`.",
    )
    return parser


def main() -> int:
    """Run the evaluation CLI."""
    parser = build_parser()
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    samples = load_dataset(dataset_dir)
    parameters = (
        load_parameters_from_json(args.params_json)
        if args.params_json
        else get_default_parameters()
    )
    report = evaluate_parameters(parameters, samples, dataset_dir=dataset_dir)
    print(f"{float(report['dataset_score']):.12f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
