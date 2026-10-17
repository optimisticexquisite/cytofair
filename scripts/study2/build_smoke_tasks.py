#!/usr/bin/env python3
"""One-epoch pipeline smoke tests run before registration (never analysed).

Only P1 (and one leak cell / one P1 adversary run) are exercised, so no smoke
run can reveal a protocol contrast. Their outputs live under ROOT/smoke and are
listed in the excluded-attempt ledger.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root, repository = Path(args.root), Path(args.repository)
    environment = [
        "env", f"PYTHONPATH={repository / 'src'}", f"TORCH_HOME={root / 'torch_home'}",
        f"HF_HOME={root / 'hf_home'}", "HF_HUB_OFFLINE=1",
    ]
    splits = repository / "splits/study2"
    probe = splits / "sipakmed_probe_split.csv"
    architectures = (
        "resnet18", "resnet50", "densenet121", "efficientnet_b0", "mobilenet_v2", "vit_b_16",
        "convnext_tiny", "swin_t", "regnet_y_1_6gf", "efficientnet_v2_s", "maxvit_t", "dinov2_vits14",
    )
    tasks = []

    def add(name, dataset, cache_name, split, protocol, architecture, extra=()):
        output = root / "smoke"
        tasks.append({
            "id": f"smoke:{name}",
            "priority": 0,
            "gpu_units": 2,
            "run_json": str(output / "runs" / f"{name}.json"),
            "command": [
                *environment, args.python, "-m", "cytofair.cli", "train-study2",
                "--cache-dir", str(root / "cache"), "--cache-name", cache_name, "--split", str(split),
                "--dataset", dataset, "--protocol", protocol, "--architecture", architecture,
                "--seed", "137", "--output", str(output), "--run-name", name, "--epochs", "1",
                "--repository", str(repository), *extra,
            ],
        })

    p1 = splits / "sipakmed/sipakmed_p1_seed137.csv"
    for architecture in architectures:
        add(f"p1_{architecture}", "SIPaKMeD", "sipakmed", p1, "P1", architecture, ("--feature-split", str(probe)))
    add("p1_resnet18_repeat", "SIPaKMeD", "sipakmed", p1, "P1", "resnet18", ("--feature-split", str(probe)))
    add(
        "leak_high_resnet50", "SIPaKMeD", "sipakmed", splits / "sipakmed_2c/sipakmed_2c_leak_high_seed137.csv",
        "leak_high", "resnet50", ("--extra-eval", f"transplants={root / 'cache'}:transplant_seed137"),
    )
    add("p1_grl_resnet50", "SIPaKMeD", "sipakmed", p1, "P1", "resnet50", ("--adversary-lambda", "1.0"))
    add("cric_p1_resnet18", "CRIC", "cric", splits / "cric/cric_p1_seed137.csv", "P1", "resnet18")
    Path(args.output).write_text("".join(json.dumps(task) + "\n" for task in tasks))
    print(f"{len(tasks)} smoke tasks")


if __name__ == "__main__":
    main()
