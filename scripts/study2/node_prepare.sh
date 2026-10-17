#!/usr/bin/env bash
# Build every node-local Study 2 input from committed manifests (no model involved).
#
#   scripts/study2/node_prepare.sh ROOT PYTHON CRIC_IMAGE_DIR
#
# ROOT is node-local storage (the training host's /tmp; the shared network home is
# too slow to feed GPUs). The SIPaKMeD crop cache must already exist in ROOT/cache.
set -euo pipefail

root=$1
python=$2
cric_images=$3
cache="$root/cache"
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=1

"$python" -m cytofair.cli build-crop-cache provenance/cric_parent_map.parquet \
  --windows-from "$cric_images" --output "$cache" --name cric --window 100

for design in splits/study2/sipakmed_2c/sipakmed_2c_design_seed*.csv; do
  seed=${design##*_seed}
  seed=${seed%.csv}
  "$python" -m cytofair.cli build-transplants "$design" \
    --cache-dir "$cache" --cache-name sipakmed --output "$cache" --name "transplant_seed$seed"
done
