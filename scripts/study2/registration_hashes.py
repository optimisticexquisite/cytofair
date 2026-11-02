#!/usr/bin/env python3
"""Write SHA-256 hashes of every file the Study 2 registration fixes.

Covers the analysis code, training and design code, task builder, split
manifest, probe split, and (when available) the node-local cache manifests, so
any later change to a registered artifact is detectable by re-hashing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REGISTERED = (
    "PREREGISTRATION_STUDY2.md",
    "analysis/study2/protocol_lmm.R",
    "analysis/study2/mechanism_lmm.R",
    "analysis/study2/power_simulation.R",
    "analysis/study2/collect_runs.py",
    "analysis/study2/prepare_mechanism.py",
    "analysis/study2/secondary.py",
    "src/cytofair/train_study2.py",
    "src/cytofair/gpu_augment.py",
    "src/cytofair/backbones.py",
    "src/cytofair/factorial.py",
    "src/cytofair/transplant.py",
    "src/cytofair/recoverability.py",
    "src/cytofair/study2_design.py",
    "src/cytofair/image_cache.py",
    "src/cytofair/fingerprint.py",
    "scripts/study2/build_tasks.py",
    "scripts/study2/launch_matrix.py",
    "splits/study2/study2_split_manifest.csv",
    "splits/study2/study2_transplant_manifest.csv",
    "splits/study2/sipakmed_probe_split.csv",
    "data_manifests/cric_images_verified.json",
)
REGISTERED_GLOBS = ("splits/study2/sipakmed_2b/background_fraction_seed*.csv", "splits/study2/tasks/*.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-manifests", nargs="*", default=[], help="node-local *_manifest.json files")
    parser.add_argument("--output", default="reports/study2/registration_hashes.txt")
    args = parser.parse_args()
    lines = []
    names = list(REGISTERED)
    for pattern in REGISTERED_GLOBS:
        names.extend(sorted(str(path) for path in Path().glob(pattern)))
    for name in names:
        lines.append(f"{hashlib.sha256(Path(name).read_bytes()).hexdigest()}  {name}")
    for manifest in sorted(args.cache_manifests):
        record = json.loads(Path(manifest).read_text())
        lines.append(f"{record['pixels_sha256']}  cache:{record['name']}:pixels ({record['n_crops']} crops)")
        lines.append(f"{record['index_sha256']}  cache:{record['name']}:index")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    print(output.read_text())


if __name__ == "__main__":
    main()
