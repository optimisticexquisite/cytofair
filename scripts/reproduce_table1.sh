#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

uv sync --locked
uv run cytofair download --manifest data_sources.json --output data/raw --connections 3
uv run cytofair extract --manifest data_sources.json --raw data/raw --output data/extracted

uv run cytofair recover-sipakmed data/extracted/sipakmed --workers 3
uv run cytofair recover-cx22 data/extracted/cx22 --image-root data/extracted/ccedd/png
uv run cytofair recover-cric data/raw/cric/classifications.csv
uv run cytofair recover-herlev data/extracted/herlev
uv run cytofair audit-duplicates provenance/sipakmed_parent_map.parquet \
  --hashes results/duplicates/sipakmed_hashes.parquet \
  --pairs results/duplicates/sipakmed_near_duplicate_pairs.csv \
  --output provenance/sipakmed_parent_map_p3.parquet
uv run cytofair make-splits provenance/sipakmed_parent_map_p3.parquet \
  --dataset SIPaKMeD --output splits/generated
uv run cytofair pixel-overlap provenance/sipakmed_branch_b.parquet --dataset SIPaKMeD

uv run cytofair table1 \
  --map SIPaKMeD=provenance/sipakmed_parent_map_p3.parquet \
  --map Cx22=provenance/cx22_parent_map.parquet \
  --map CRIC=provenance/cric_parent_map.parquet \
  --map Herlev=provenance/herlev_parent_map.parquet \
  --output results/table1_structural_leakage.csv \
  --seed-output results/table1_seed_level.csv
