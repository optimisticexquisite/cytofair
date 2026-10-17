#!/usr/bin/env bash
# Run the registered Study 2 analysis end to end.
#
#   analysis/study2/run_all.sh ROOT PYTHON RSCRIPT [OUTPUT]
#
# ROOT holds the run outputs (ROOT/runs/<experiment>/...). The collector fails
# closed on any missing or mismatched registered run, so nothing downstream can
# summarise an incomplete matrix.
set -euo pipefail

root=$1
python=$2
rscript=$3
output=${4:-results/study2/analysis}
tasks=splits/study2/tasks
mkdir -p "$output"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

"$python" analysis/study2/collect_runs.py \
  "$tasks/fingerprint.jsonl" "$tasks/phase1.jsonl" "$tasks/factorial.jsonl" "$tasks/grl.jsonl" "$tasks/cric.jsonl" \
  --output "$output/runs"

# H1 (primary) and its registered secondary contrasts.
"$rscript" analysis/study2/protocol_lmm.R "$output/runs/phase1_runs.csv" "$output/h1_accuracy.json" accuracy P1 P2 12 20
"$rscript" analysis/study2/protocol_lmm.R "$output/runs/phase1_runs.csv" "$output/p2_p3_accuracy.json" accuracy P2 P3 12 20
"$rscript" analysis/study2/protocol_lmm.R "$output/runs/phase1_runs.csv" "$output/h1_macro_f1.json" macro_f1 P1 P2 12 20
"$rscript" analysis/study2/protocol_lmm.R "$output/runs/phase1_runs.csv" "$output/h1_balanced_accuracy.json" balanced_accuracy P1 P2 12 20

# H2 (CRIC) and its secondary contrasts.
"$rscript" analysis/study2/protocol_lmm.R "$output/runs/cric_runs.csv" "$output/h2_cric_accuracy.json" accuracy P1 P2 6 10
"$rscript" analysis/study2/protocol_lmm.R "$output/runs/cric_runs.csv" "$output/cric_p1prime_accuracy.json" accuracy P1-prime P2 6 10
"$rscript" analysis/study2/protocol_lmm.R "$output/runs/cric_runs.csv" "$output/cric_balanced_accuracy.json" balanced_accuracy P1 P2 6 10
"$rscript" analysis/study2/protocol_lmm.R "$output/runs/cric_runs.csv" "$output/cric_macro_f1.json" macro_f1 P1 P2 6 10

# Mechanism arms (H3-H5) and the Holm family with H2.
"$python" analysis/study2/prepare_mechanism.py --runs "$output/runs" --output "$output/mechanism" --workers 4
"$rscript" analysis/study2/mechanism_lmm.R \
  "$output/runs/factorial_runs.csv" "$output/mechanism/transplant_runs.csv" \
  "$output/mechanism/grl_recoverability.csv" "$output/h2_cric_accuracy.json" "$output/mechanism.json"

# Registered secondary analyses.
"$python" analysis/study2/secondary.py --runs "$output/runs" --mechanism "$output/mechanism" \
  --fingerprint "$root/runs/fingerprint/fingerprint_embeddings.npz" \
  --h1 "$output/h1_accuracy.json" --mechanism-json "$output/mechanism.json" \
  --output "$output/secondary.json"
