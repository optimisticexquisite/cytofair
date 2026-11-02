#!/usr/bin/env bash
# Run the registered Amendment 8 follow-up analysis end to end.
#
#   analysis/study2/followup/run_followup.sh ROOT PYTHON RSCRIPT [OUTPUT]
#
# ROOT holds the run outputs (ROOT/runs/followup/...). The preparer fails closed
# on any missing or mismatched registered task, so nothing downstream can
# summarise an incomplete arm. The Study 2 collector tables
# (results/study2/analysis/runs) must already exist.
set -euo pipefail

root=$1
python=$2
rscript=$3
output=${4:-results/study2/followup}
mkdir -p "$output"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

"$python" analysis/study2/followup/cric_context.py \
  --runs results/study2/analysis/runs/cric_runs.csv --output "$output/cric_context.json"
"$python" analysis/study2/followup/prepare_followup.py \
  --tasks splits/study2/followup/tasks --logs "$root/runs/followup/logs" --output "$output/tables"
"$rscript" analysis/study2/followup/followup_lmm.R "$output/tables" "$output/cric_context.json" "$output/followup.json"
