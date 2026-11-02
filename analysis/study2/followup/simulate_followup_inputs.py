#!/usr/bin/env python3
"""Synthetic follow-up tables with planted effects, for dry-running ``followup_lmm.R``.

The planted values are arbitrary; they exist only to check that every registered
model fits and that each estimate recovers its sign and rough size.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FACTORIAL = ("resnet50", "densenet121", "convnext_tiny", "vit_b_16")
ALL = ("resnet18", "resnet50", "densenet121", "efficientnet_b0", "mobilenet_v2", "vit_b_16",
       "convnext_tiny", "swin_t", "regnet_y_1_6gf", "efficientnet_v2_s", "maxvit_t", "dinov2_vits14")
CELLS = ("leak_high", "clean_high", "leak_low", "clean_low")
ABLATIONS = ("original", "grey", "stain_transfer", "low_pass", "illumination_flat", "background_grey")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    rows, cams = [], []
    for seed in range(10):
        for arch in FACTORIAL:
            for cell in CELLS:
                leak = cell.startswith("leak")
                base = 0.95 + 0.017 * leak + rng.normal(0, 0.004)
                for ablation in ABLATIONS:
                    drop = {"original": 0, "grey": 0.05, "stain_transfer": 0.03, "low_pass": 0.02,
                            "illumination_flat": 0.01, "background_grey": 0.02}[ablation]
                    removed = 0.012 * leak if ablation == "stain_transfer" else 0.0
                    rows.append({"architecture": arch, "seed": seed, "cell": cell, "ablation": ablation,
                                 "accuracy": base - drop - removed + rng.normal(0, 0.003), "crops": 527,
                                 "original_prediction_agreement": 0.999})
                if arch != "vit_b_16":
                    cams.append({"architecture": arch, "seed": seed, "cell": cell,
                                 "background_attribution_fraction": 0.2 + 0.01 * leak + rng.normal(0, 0.01),
                                 "background_pixel_fraction": 0.25, "random_head_spearman": 0.3, "crops": 527})
    pd.DataFrame(rows).to_csv(out / "ablation_runs.csv", index=False)
    pd.DataFrame(cams).to_csv(out / "gradcam_runs.csv", index=False)

    carrier = []
    for seed in range(10):
        for condition, leak_effect in (("standard", 0.017), ("masked", 0.015), ("colour", 0.006)):
            for arch in FACTORIAL:
                for cell in CELLS:
                    carrier.append({"architecture": arch, "seed": 2000 + seed, "protocol": cell, "cell": cell,
                                    "condition": condition,
                                    "accuracy": 0.95 + leak_effect * cell.startswith("leak") + rng.normal(0, 0.004)})
    pd.DataFrame(carrier).to_csv(out / "carrier_runs.csv", index=False)

    leace = []
    for seed in range(20):
        for arch in ALL:
            for protocol in ("P1", "P2"):
                for rank, removed in zip((0, 4, 16, 64, 256), (0, 0.2, 0.4, 0.6, 0.8), strict=True):
                    gap = 0.01 * (1 - removed) if protocol == "P1" else 0.0
                    leace.append({"architecture": arch, "seed": seed, "protocol": protocol, "rank": rank,
                                  "accuracy": 0.96 + gap + rng.normal(0, 0.004),
                                  "within_class_accuracy": 0.25 * (1 - removed) + 0.02 + rng.normal(0, 0.01),
                                  "within_class_floor": 0.0118, "original_head_accuracy": 0.96,
                                  "recorded_accuracy": 0.96})
    pd.DataFrame(leace).to_csv(out / "leace_runs.csv", index=False)

    normal = []
    for seed in range(10):
        for arch in ("resnet50", "convnext_tiny", "densenet121", "efficientnet_b0"):
            for strength in (0.0, 0.5, 1.0):
                for protocol in ("P1", "P2"):
                    gap = 0.01 * (1 - 0.5 * strength) if protocol == "P1" else 0.0
                    normal.append({"architecture": arch, "seed": 600 + seed, "protocol": protocol,
                                   "strength": strength, "accuracy": 0.96 + gap + rng.normal(0, 0.004),
                                   "within_class_accuracy": 0.25 - 0.08 * strength + rng.normal(0, 0.01),
                                   "within_class_floor": 0.0118})
    pd.DataFrame(normal).to_csv(out / "normalization_recoverability.csv", index=False)

    linear = []
    for seed in range(10):
        for arch in ("resnet50", "convnext_tiny", "densenet121"):
            for lam, removed in zip((0.0, 0.1, 0.3, 1.0, 3.0), (0, 0.2, 0.4, 0.5, 0.6), strict=True):
                for protocol in ("P1", "P2"):
                    gap = 0.01 * (1 - removed) if protocol == "P1" else 0.0
                    linear.append({"architecture": arch, "seed": 600 + seed, "protocol": protocol,
                                   "adversary_lambda": lam, "accuracy": 0.96 + gap + rng.normal(0, 0.004),
                                   "within_class_accuracy": 0.25 * (1 - removed) + rng.normal(0, 0.01),
                                   "within_class_floor": 0.0118})
    pd.DataFrame(linear).to_csv(out / "linadv_recoverability.csv", index=False)
    pd.DataFrame(columns=["task_id", "architecture", "adversary_lambda", "diverged_at_epoch"]).to_csv(
        out / "linadv_diverged.csv", index=False)
    (out / "cric_context.json").write_text(json.dumps({"F1": {"estimate": 2.0, "p_two_sided": 0.001,
                                                              "ci95": [1.0, 3.0]}}))
    print(f"wrote synthetic follow-up tables to {out}")


if __name__ == "__main__":
    main()
