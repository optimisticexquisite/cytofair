#!/usr/bin/env bash
# Copy one Study 2 experiment's outputs to static storage, verify, then free its node-local checkpoints.
#
#   scripts/study2/offload_checkpoints.sh ROOT ARCHIVE EXPERIMENT PYTHON
#
# rsync verifies every transferred file with a whole-file checksum; on top of that
# every archived checkpoint must match its size and a seeded sample of 50 is
# re-hashed against the SHA-256 recorded by its training run. Only then are the
# node-local checkpoints (the bulk of the space) deleted; predictions, features,
# and run records stay in place for the analyses.
set -euo pipefail

root=$1
archive=$2
experiment=$3
python=$4
mkdir -p "$archive/runs"
rsync -a --bwlimit=20000 "$root/runs/$experiment" "$archive/runs/"

"$python" - "$root" "$archive" "$experiment" <<'EOF'
import hashlib, json, random, sys
from pathlib import Path

root, archive, experiment = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
checkpoints = []
for run in sorted((root / "runs" / experiment / "runs").glob("*.json")):
    item = json.loads(run.read_text())["artifacts"]["checkpoint"]
    local = Path(item["path"])
    copied = archive / local.relative_to(root)
    if not copied.is_file() or copied.stat().st_size != local.stat().st_size:
        raise SystemExit(f"archive copy missing or truncated: {copied}")
    checkpoints.append((copied, item["sha256"]))
random.Random(20261019).shuffle(checkpoints)
for path, expected in checkpoints[:50]:
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"hash mismatch: {path}")
print(f"{experiment}: {len(checkpoints)} checkpoints archived, sizes match, 50 re-hashed OK")
EOF

rm -rf "$root/runs/$experiment/checkpoints"
echo "freed $root/runs/$experiment/checkpoints"
