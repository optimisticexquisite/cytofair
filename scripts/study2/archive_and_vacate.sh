#!/usr/bin/env bash
# Archive Study 2 outputs to static storage, verify them, then vacate node-local space.
#
#   scripts/study2/archive_and_vacate.sh ROOT ARCHIVE PYTHON
#
# The network home is slow and can stall under load, so the copy is a single
# bandwidth-limited rsync stream. Every run JSON already records the SHA-256 of
# each artifact; after copying, a sample of archived artifacts is re-hashed
# against those values, and ROOT is deleted only if every check passes.
set -euo pipefail

root=$1
archive=$2
python=$3
mkdir -p "$archive"

rsync -a --bwlimit=40000 --exclude 'cache/' --exclude 'pkgs/' --exclude 'envs/' \
  --exclude 'torch_home/' --exclude 'hf_home/' --exclude 'data/cric_images/' \
  "$root/runs" "$root/smoke" "$root/logs" "$archive/"

"$python" - "$root" "$archive" <<'EOF'
import hashlib, json, random, sys
from pathlib import Path

root, archive = Path(sys.argv[1]), Path(sys.argv[2])
runs = sorted((archive / "runs").glob("*/runs/*.json"))
artifacts = []
for run in runs:
    record = json.loads(run.read_text())
    for item in record.get("artifacts", {}).values():
        archived = archive / Path(item["path"]).relative_to(root)
        if not archived.is_file():
            raise SystemExit(f"missing archived artifact: {archived}")
        artifacts.append((archived, item["sha256"]))
random.Random(20261018).shuffle(artifacts)
for path, expected in artifacts[:200]:
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"hash mismatch: {path}")
print(f"{len(runs)} run records, {len(artifacts)} artifacts present; 200 re-hashed OK")
EOF

rm -rf "$root" /dev/shm/rchethan1_cytofair
echo "vacated $root and /dev/shm/rchethan1_cytofair"
