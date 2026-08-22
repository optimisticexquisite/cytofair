#!/usr/bin/env python3
"""Verify the preregistered SIPaKMeD classification matrix before summarising it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cytofair.results import verify_classification_matrix

ARCHITECTURES = (
    "resnet18",
    "resnet50",
    "densenet121",
    "efficientnet_b0",
    "mobilenet_v2",
    "vit_b_16",
)
PROTOCOLS = ("P1", "P1-prime", "P2", "P3")
SEEDS = (17, 29, 43, 71, 101)
EXPECTED_RUNTIME = {
    "numpy_version": "2.5.2",
    "pandas_version": "3.0.5",
    "scipy_version": "1.18.1",
    "scikit_learn_version": "1.9.0",
    "pillow_version": "12.3.0",
    "opencv_version": "5.0.0",
    "torch_version": "2.9.0+cu128",
    "torchvision_version": "0.24.0+cu128",
    "cuda_version": "12.8",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/classification")
    parser.add_argument("--splits", default="splits/generated")
    parser.add_argument("--git-sha", required=True)
    parser.add_argument(
        "--output", default="reports/generated/classification_matrix_validation.json"
    )
    args = parser.parse_args()
    result_paths = sorted(Path(args.results).glob("*.json"))
    evidence = verify_classification_matrix(
        result_paths,
        split_root=args.splits,
        dataset="SIPaKMeD",
        architectures=ARCHITECTURES,
        protocols=PROTOCOLS,
        seeds=SEEDS,
        expected_git_sha=args.git_sha,
        expected_runtime=EXPECTED_RUNTIME,
        expected_train_crops=3239,
        expected_test_crops=810,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))
    if not evidence["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
