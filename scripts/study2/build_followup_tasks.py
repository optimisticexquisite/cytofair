#!/usr/bin/env python3
"""Write the Amendment 8 follow-up matrix as JSONL task files for ``launch_matrix.py``.

Evaluation-only tasks reuse saved Study 2 checkpoints; training tasks use fresh
seeds (carrier factorial) or the Study 2 seeds no mechanism arm used (natural
P1/P2 splits). Every task names its split and SHA-256 from a committed manifest,
and the collector re-checks the fields listed under ``expect``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

FACTORIAL_ARCHITECTURES = ("resnet50", "densenet121", "convnext_tiny", "vit_b_16")
ALL_ARCHITECTURES = (
    "resnet18", "resnet50", "densenet121", "efficientnet_b0", "mobilenet_v2", "vit_b_16",
    "convnext_tiny", "swin_t", "regnet_y_1_6gf", "efficientnet_v2_s", "maxvit_t", "dinov2_vits14",
)
CARRIER_CONDITIONS = {
    "standard": {"cache": "sipakmed", "color": None},
    "masked": {"cache": "sipakmed_masked", "color": None},
    "colour": {"cache": "sipakmed", "color": "hed_strong_bc"},
}
NORMALIZATION_ARCHITECTURES = ("resnet50", "convnext_tiny", "densenet121", "efficientnet_b0")
NORMALIZATION_STRENGTHS = (50, 100)
LINEAR_ADVERSARY_ARCHITECTURES = ("resnet50", "convnext_tiny", "densenet121")
LINEAR_ADVERSARY_LAMBDAS = (0.0, 0.1, 0.3, 1.0, 3.0)
STUDY2_SEEDS = (137, 173, 211, 257, 307, 349, 401, 457, 503, 557,
                601, 653, 701, 751, 809, 853, 907, 953, 1009, 1051)
MECHANISM_SEEDS = STUDY2_SEEDS[:10]
FOLLOWUP_SPLIT_SEEDS = STUDY2_SEEDS[10:]
FOLLOWUP_FACTORIAL_SEEDS = (1103, 1151, 1201, 1249, 1301, 1361, 1409, 1451, 1499, 1553)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study2-manifest", default="splits/study2/study2_split_manifest.csv")
    parser.add_argument("--followup-manifest", default="splits/study2/followup/followup_split_manifest.csv")
    parser.add_argument("--root", required=True, help="node-local working root")
    parser.add_argument("--repository", required=True, help="checkout at the Amendment 8 registration commit")
    parser.add_argument("--study2-runs", required=True, help="directory holding the Study 2 run outputs")
    parser.add_argument("--python", required=True)
    parser.add_argument("--output", default="splits/study2/followup/tasks")
    args = parser.parse_args()

    root, repository, study2_runs = Path(args.root), Path(args.repository), Path(args.study2_runs)
    cache = root / "cache"
    environment = [
        "env", f"PYTHONPATH={repository / 'src'}", f"TORCH_HOME={root / 'torch_home'}",
        f"HF_HOME={root / 'hf_home'}", "HF_HUB_OFFLINE=1",
    ]
    cli = [*environment, args.python, "-m", "cytofair.cli"]
    probe_split = repository / "splits/study2/sipakmed_probe_split.csv"
    study2 = pd.read_csv(args.study2_manifest)
    followup = pd.read_csv(args.followup_manifest)
    out_root = root / "runs" / "followup"

    def split_row(manifest, dataset, protocol, seed):
        rows = manifest.loc[(manifest.dataset == dataset) & (manifest.protocol == protocol) & (manifest.seed == seed)]
        if len(rows) != 1:
            raise SystemExit(f"split {dataset}/{protocol}/{seed} not uniquely registered")
        return rows.iloc[0]

    files: dict[str, list[dict]] = {name: [] for name in ("ffact", "leace", "carrier", "linadv", "normalization")}

    # Evaluation-only: carrier ablations + Grad-CAM on every saved 2C model.
    for seed in MECHANISM_SEEDS:
        for architecture in FACTORIAL_ARCHITECTURES:
            for cell in ("leak_high", "clean_high", "leak_low", "clean_low"):
                name = f"sipakmed_{architecture}_{cell}_seed{seed}"
                row = split_row(study2, "SIPaKMeD", cell, seed)
                output = out_root / "factorial_eval"
                files["ffact"].append({
                    "id": f"ffact:{name}", "experiment": "ffact", "priority": 0, "gpu_units": 1,
                    "command": [*cli, "followup-factorial", str(study2_runs / "factorial" / "runs" / f"{name}.json"),
                                "--cache-dir", str(cache), "--cache-name", "sipakmed",
                                "--ablation-dir", str(cache), "--ablation-name", f"ablation_seed{seed}",
                                "--output", str(output), "--repository", str(repository)],
                    "run_json": str(output / "runs" / f"{name}.json"),
                    "expect": {"architecture": architecture, "protocol": cell, "seed": seed,
                               "split_file_sha256": row.sha256},
                })

    # Evaluation-only: LEACE erasure ladder on every saved Phase 1 P1/P2 model.
    for seed in STUDY2_SEEDS:
        for architecture in ALL_ARCHITECTURES:
            for protocol in ("P1", "P2"):
                name = f"sipakmed_{architecture}_{protocol.lower()}_seed{seed}"
                row = split_row(study2, "SIPaKMeD", protocol, seed)
                output = out_root / "leace"
                files["leace"].append({
                    "id": f"leace:{name}", "experiment": "leace", "priority": 1, "gpu_units": 1,
                    "command": [*cli, "followup-leace", str(study2_runs / "phase1" / "runs" / f"{name}.json"),
                                "--cache-dir", str(cache), "--cache-name", "sipakmed",
                                "--probe-split", str(probe_split), "--output", str(output),
                                "--repository", str(repository)],
                    "run_json": str(output / "runs" / f"{name}.json"),
                    "expect": {"architecture": architecture, "protocol": protocol, "seed": seed,
                               "split_file_sha256": row.sha256},
                })

    def train(experiment, priority, manifest, protocol, architecture, seed, *, suffix, cache_name, extra=()):
        row = split_row(manifest, "SIPaKMeD", protocol, seed)
        safe = protocol.lower().replace("-", "_")
        name = f"sipakmed_{architecture}_{safe}_seed{seed}_{suffix}"
        output = out_root / experiment
        command = [*cli, "train-study2", "--cache-dir", str(cache), "--cache-name", cache_name,
                   "--split", str(repository / row.path), "--split-sha256", row.sha256,
                   "--dataset", "SIPaKMeD", "--protocol", protocol, "--architecture", architecture,
                   "--seed", str(seed), "--output", str(output), "--run-name", name,
                   "--repository", str(repository), "--feature-split", str(probe_split), *extra]
        return {
            "id": f"{experiment}:{name}", "experiment": experiment, "priority": priority, "gpu_units": 2,
            "command": command, "run_json": str(output / "runs" / f"{name}.json"),
            "expect": {"dataset": "SIPaKMeD", "protocol": protocol, "architecture": architecture, "seed": seed,
                       "split_file_sha256": row.sha256, "cache_name": cache_name},
        }

    # Seed-blocked order, so an interruption leaves complete seeds.
    for seed in FOLLOWUP_FACTORIAL_SEEDS:
        for condition, spec in CARRIER_CONDITIONS.items():
            for architecture in FACTORIAL_ARCHITECTURES:
                for cell in ("leak_high", "clean_high", "leak_low", "clean_low"):
                    extra = () if spec["color"] is None else ("--color-augment", spec["color"])
                    task = train("carrier", 2, followup, cell, architecture, seed, suffix=condition,
                                 cache_name=spec["cache"], extra=extra)
                    task["expect"]["color_augment"] = spec["color"]
                    files["carrier"].append(task)
    for seed in FOLLOWUP_SPLIT_SEEDS:
        for architecture in LINEAR_ADVERSARY_ARCHITECTURES:
            for adversary in LINEAR_ADVERSARY_LAMBDAS:
                for protocol in ("P1", "P2"):
                    extra = ("--adversary-lambda", repr(adversary), "--adversary-kind", "linear_bn",
                             "--adversary-reset-epochs", "5", "--adversary-learning-rate", "0.001")
                    task = train("linadv", 3, study2, protocol, architecture, seed, suffix=f"linadv{adversary:g}",
                                 cache_name="sipakmed", extra=extra)
                    task["expect"].update(adversary_lambda=adversary, adversary_kind="linear_bn")
                    files["linadv"].append(task)
    for seed in FOLLOWUP_SPLIT_SEEDS:
        for architecture in NORMALIZATION_ARCHITECTURES:
            for strength in NORMALIZATION_STRENGTHS:
                for protocol in ("P1", "P2"):
                    files["normalization"].append(
                        train("normalization", 4, study2, protocol, architecture, seed, suffix=f"reinhard{strength}",
                              cache_name=f"sipakmed_reinhard{strength}")
                    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for name, tasks in files.items():
        (output / f"{name}.jsonl").write_text("".join(json.dumps(item) + "\n" for item in tasks))
        print(f"{name}: {len(tasks)} tasks")


if __name__ == "__main__":
    main()
