#!/usr/bin/env python3
"""Append the SHA-256 hashes fixed by Amendment 8 to the Study 2 registration report.

Covers the amendment, every follow-up code, design, and task file, and (from
their manifests) the derived node-local caches the follow-up arms read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

FILES = (
    "PREREGISTRATION_STUDY2.md",
    "PREREGISTRATION_STUDY2_AMENDMENT8.md",
    "src/cytofair/carriers.py",
    "src/cytofair/followup_eval.py",
    "src/cytofair/train_study2.py",
    "src/cytofair/gpu_augment.py",
    "src/cytofair/backbones.py",
    "src/cytofair/cli.py",
    "src/cytofair/study2_design.py",
    "scripts/study2/build_followup_tasks.py",
    "scripts/study2/launch_matrix.py",
    "splits/study2/followup/followup_split_manifest.csv",
    "splits/study2/followup/followup_ablation_manifest.csv",
)
GLOBS = (
    "analysis/study2/followup/*",
    "splits/study2/followup/ablations/*.csv",
    "splits/study2/followup/tasks/*.jsonl",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-manifests", nargs="*", default=[])
    parser.add_argument("--report", default="reports/study2/registration_hashes.txt")
    parser.add_argument("--date", required=True)
    args = parser.parse_args()
    names = list(FILES)
    for pattern in GLOBS:
        names.extend(sorted(str(path) for path in Path().glob(pattern) if path.is_file()))
    lines = ["", f"# Amendment 8 ({args.date}): follow-up experiments, registered before any follow-up run or analysis"]
    lines += [f"{hashlib.sha256(Path(name).read_bytes()).hexdigest()}  {name}" for name in names]
    for manifest in sorted(args.cache_manifests):
        record = json.loads(Path(manifest).read_text())
        lines.append(f"{record['pixels_sha256']}  cache:{record['name']}:pixels ({record['n_crops']} crops)")
        lines.append(f"{record['index_sha256']}  cache:{record['name']}:index")
    with Path(args.report).open("a") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
