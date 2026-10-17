#!/usr/bin/env python3
"""Write the registered Study 2 run matrix as JSONL task files for ``launch_matrix.py``.

Every task names the exact split file and its SHA-256 from the committed split
manifest, the output paths, and the fields its result must carry. The collector
re-checks those fields, so a run trained on any other split cannot enter the
analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ALL_ARCHITECTURES = (
    "resnet18", "resnet50", "densenet121", "efficientnet_b0", "mobilenet_v2", "vit_b_16",
    "convnext_tiny", "swin_t", "regnet_y_1_6gf", "efficientnet_v2_s", "maxvit_t", "dinov2_vits14",
)
FACTORIAL_ARCHITECTURES = ("resnet50", "densenet121", "convnext_tiny", "vit_b_16")
GRL_ARCHITECTURES = ("resnet50", "convnext_tiny")
GRL_LAMBDAS = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
CRIC_ARCHITECTURES = ("resnet18", "resnet50", "densenet121", "efficientnet_b0", "convnext_tiny", "vit_b_16")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="splits/study2/study2_split_manifest.csv")
    parser.add_argument("--root", required=True, help="node-local working root")
    parser.add_argument("--repository", required=True, help="checkout at the registered commit")
    parser.add_argument("--python", required=True)
    parser.add_argument("--seeds", type=int, required=True, help="number of Phase 1 seeds (power analysis)")
    parser.add_argument("--output", default="splits/study2/tasks")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    root = Path(args.root)
    cache = root / "cache"
    repository = Path(args.repository)
    environment = [
        "env",
        f"PYTHONPATH={repository / 'src'}",
        f"TORCH_HOME={root / 'torch_home'}",
        f"HF_HOME={root / 'hf_home'}",
        "HF_HUB_OFFLINE=1",
    ]
    probe_split = repository / "splits/study2/sipakmed_probe_split.csv"
    phase1_seeds = sorted(manifest.loc[manifest.protocol == "P1"].query("dataset == 'SIPaKMeD'").seed.unique())
    registered_seeds = [137, 173, 211, 257, 307, 349, 401, 457, 503, 557,
                        601, 653, 701, 751, 809, 853, 907, 953, 1009, 1051]
    if phase1_seeds != sorted(registered_seeds):
        raise SystemExit("manifest seeds differ from the registered Study 2 seed list")
    phase1_seeds = registered_seeds[: args.seeds]
    mechanism_seeds = registered_seeds[:10]

    def split_row(dataset: str, protocol: str, seed: int) -> pd.Series:
        rows = manifest.loc[(manifest.dataset == dataset) & (manifest.protocol == protocol) & (manifest.seed == seed)]
        if len(rows) != 1:
            raise SystemExit(f"split {dataset}/{protocol}/{seed} not uniquely registered")
        return rows.iloc[0]

    def task(experiment, priority, dataset, cache_name, protocol, architecture, seed, *, extra=(), adversary=None,
             features=True):
        row = split_row(dataset, protocol, seed)
        suffix = "" if adversary is None else f"_lambda{adversary:g}"
        safe_protocol = protocol.lower().replace("-", "_")
        name = f"{dataset.lower()}_{architecture}_{safe_protocol}_seed{seed}{suffix}"
        output = root / "runs" / experiment
        command = [
            *environment, args.python, "-m", "cytofair.cli", "train-study2",
            "--cache-dir", str(cache), "--cache-name", cache_name,
            "--split", str(repository / row.path), "--split-sha256", row.sha256,
            "--dataset", dataset, "--protocol", protocol, "--architecture", architecture,
            "--seed", str(seed), "--output", str(output), "--run-name", name,
            "--repository", str(repository),
        ]
        if features:
            command += ["--feature-split", str(probe_split)]
        if adversary is not None:
            command += ["--adversary-lambda", repr(float(adversary))]
        for item in extra:
            command += ["--extra-eval", item]
        return {
            "id": f"{experiment}:{name}",
            "experiment": experiment,
            "priority": priority,
            "gpu_units": 2,
            "command": command,
            "run_json": str(output / "runs" / f"{name}.json"),
            "expect": {
                "dataset": dataset,
                "protocol": protocol,
                "architecture": architecture,
                "seed": int(seed),
                "split_file_sha256": row.sha256,
                "adversary_lambda": None if adversary is None else float(adversary),
            },
        }

    files: dict[str, list[dict]] = {name: [] for name in ("fingerprint", "phase1", "factorial", "grl", "cric")}
    fingerprint_output = root / "runs" / "fingerprint"
    files["fingerprint"].append({
        "id": "fingerprint:resnet18_probe_seed17",
        "experiment": "fingerprint",
        "priority": 0,
        "gpu_units": 2,
        "command": [
            *environment, args.python, "-m", "cytofair.cli", "train-fingerprint",
            str(repository / "provenance/sipakmed_parent_map_p3.parquet"),
            "--cache-dir", str(cache), "--cache-name", "sipakmed", "--output", str(fingerprint_output),
            "--seed", "17", "--repository", str(repository),
        ],
        "run_json": str(fingerprint_output / "fingerprint_encoder.json"),
        "expect": {"task": "parent_fingerprint_encoder", "seed": 17},
    })
    # Seed-blocked order: an interruption leaves complete seeds, never a ragged matrix.
    for seed in phase1_seeds:
        for architecture in ALL_ARCHITECTURES:
            for protocol in ("P1", "P2", "P3"):
                files["phase1"].append(task("phase1", 1, "SIPaKMeD", "sipakmed", protocol, architecture, seed))
    for seed in mechanism_seeds:
        transplants = f"transplants={cache}:transplant_seed{seed}"
        for architecture in FACTORIAL_ARCHITECTURES:
            for cell in ("leak_high", "clean_high", "leak_low", "clean_low"):
                files["factorial"].append(
                    task("factorial", 2, "SIPaKMeD", "sipakmed", cell, architecture, seed, extra=(transplants,))
                )
    for seed in mechanism_seeds:
        for architecture in GRL_ARCHITECTURES:
            for adversary in GRL_LAMBDAS:
                for protocol in ("P1", "P2"):
                    files["grl"].append(
                        task("grl", 3, "SIPaKMeD", "sipakmed", protocol, architecture, seed, adversary=adversary)
                    )
    for seed in mechanism_seeds:
        for architecture in CRIC_ARCHITECTURES:
            for protocol in ("P1", "P1-prime", "P2"):
                files["cric"].append(
                    task("cric", 4, "CRIC", "cric", protocol, architecture, seed, features=False)
                )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for name, tasks in files.items():
        (output / f"{name}.jsonl").write_text("".join(json.dumps(item) + "\n" for item in tasks))
        print(f"{name}: {len(tasks)} tasks")


if __name__ == "__main__":
    main()
