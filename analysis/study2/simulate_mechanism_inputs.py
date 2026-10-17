#!/usr/bin/env python3
"""Synthetic inputs with known effects for dry-running the registered mechanism models.

Used only to prove, before any Study 2 result exists, that ``mechanism_lmm.R``
runs end to end and recovers planted effects. Nothing here is evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    architectures = ("resnet50", "densenet121", "convnext_tiny", "vit_b_16")
    seeds = (137, 173, 211, 257, 307, 349, 401, 457, 503, 557)
    cells = ("leak_high", "clean_high", "leak_low", "clean_low")
    factorial, transplant = [], []
    for architecture in architectures:
        arch_effect = rng.normal(0, 0.5)
        for seed in seeds:
            seed_effect = rng.normal(0, 0.5)
            for cell in cells:
                leak = 0.5 if cell.startswith("leak") else -0.5
                div = 0.5 if cell.endswith("high") else -0.5
                accuracy = 0.93 + (1.2 * leak + 0.4 * div + arch_effect + seed_effect + rng.normal(0, 0.6)) / 100
                factorial.append({"architecture": architecture, "seed": seed, "protocol": cell, "accuracy": accuracy})
                route = (0.9 * leak + 0.2 + rng.normal(0, 0.5)) / 100
                for subset in ("all", "background_ge_0.30"):
                    transplant.append({
                        "architecture": architecture, "seed": seed, "cell": cell, "accuracy": accuracy,
                        "subset": subset, "recipients": 520,
                        "acc_original": accuracy, "acc_self": accuracy - 0.002,
                        "acc_sibling": accuracy - 0.01, "acc_unseen_same_class": accuracy - 0.01 - route,
                        "acc_unseen_other_class": accuracy - 0.05,
                        "donor_probability_shift": 0.03 + rng.normal(0, 0.01), "flip_to_donor_rate": 0.02,
                    })
    grl = []
    lambdas = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
    for architecture in ("resnet50", "convnext_tiny"):
        for seed in seeds:
            for index, adversary in enumerate(lambdas):
                recoverability = 0.30 - 0.04 * index
                degradation = 0.03 if adversary >= 3.0 else 0.0
                for protocol in ("P1", "P2"):
                    gap = 0.05 * (recoverability - 0.012) if protocol == "P1" else 0.0
                    grl.append({
                        "architecture": architecture, "seed": seed, "protocol": protocol,
                        "adversary_lambda": adversary,
                        "accuracy": 0.95 + gap - degradation + rng.normal(0, 0.005),
                        "within_class_accuracy": recoverability + rng.normal(0, 0.01),
                        "within_class_floor": 0.012,
                    })
    pd.DataFrame(factorial).to_csv(output / "factorial_runs.csv", index=False)
    pd.DataFrame(transplant).to_csv(output / "transplant_runs.csv", index=False)
    pd.DataFrame(grl).to_csv(output / "grl_recoverability.csv", index=False)
    (output / "cric_lmm.json").write_text(json.dumps(
        {"primary_estimate": 2.0, "primary_p": 0.001, "primary_source": "lmm"}
    ))


if __name__ == "__main__":
    main()
