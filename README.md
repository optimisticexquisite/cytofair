# CytoFair

CytoFair recovers crop-to-parent provenance, measures structural train/test leakage,
generates parent-disjoint splits, and runs preregistered statistical checks for public
cervical-cytology benchmarks.

The repository is being built from the acceptance criteria in
[PREREGISTRATION.md](PREREGISTRATION.md). Raw images are never committed or
redistributed. Download commands point to dataset owners' releases and retain source
checksums and terms.

## Quick start

```bash
uv sync --extra dev
uv run pytest
uv run cytofair --help
```

The eventual clean-clone reproduction entry point is:

```bash
./scripts/reproduce_table1.sh
```

It is deliberately GPU-free: Table 1 contains structural leakage measurements. Model
experiments and checkpoints are a separate, explicitly invoked pipeline.

## Split files

Generated split files contain `crop_id`, `parent_id`, `partition`, `protocol`,
and `seed`. Downstream code should join on `crop_id` and must never independently
resplit released crops.

Create the five fixed confirmatory seeds with:

```bash
uv run cytofair make-splits provenance/sipakmed_parent_map_p3.parquet \
  --dataset SIPaKMeD --output splits
```

P1 is a stratified crop split; P1-prime retains P1's test set while matching the P2
training count; P2 groups by the recovered parent; P3 groups whole connected
components after the cross-parent near-duplicate audit. Every CSV has a SHA-256 in
the adjacent manifest.

## Verification contract

The analysis fails closed. Provenance is not accepted unless the two independent
SIPaKMeD branches agree on at least 99% of common crops, every image-matched bbox is
inside its parent, at least 95% of image matches clear the preregistered
confidence/margin gate, the dataset total is 4,049, and the five parent counts are
126/108/238/271/223 in release order. Ambiguous rows remain in the map with
`accepted=false`; they are never silently discarded.

Cx22 source identities and requested ROIs come directly from the authors' MATLAB
generator metadata. The clean reproduction additionally downloads the declared
CCEDD archive, verifies its checksum, checks that each source image exists, and
clips the rare edge-touching ROI exactly as the generator code does. The public
Cx22 archives contain labels and generator inputs rather than a separately released
RGB `ImageDataSet.mat`, so there is no independent released crop image on which to
run a second content-matching branch. The “A/B where both exist” criterion is
therefore evaluated on SIPaKMeD; Cx22 receives metadata, source-existence, geometry,
and subset-overlap checks.

The final verification report uses explicit `PASS`, `FAIL`, and `PENDING` states.
Missing evidence can never be interpreted as a pass.

## GPU experiments

The parent-identity probe and matched classifier are intentionally separate from
Table 1. Launchers query `nvidia-smi`, accept only devices with at most 512 MiB used
and at most 5% utilization, and hard-cap concurrency at three GPUs:

```bash
python scripts/launch_probe_matrix.py provenance/sipakmed_parent_map_p3.parquet \
  results/probe --max-gpus 3 \
  --modes rgb grayscale stain_normalized background_only highpass

python scripts/launch_classification_matrix.py \
  --dataset-map SIPaKMeD=provenance/sipakmed_parent_map_p3.parquet \
  --output results/classification --max-gpus 3
```

Every model JSON records the seed, input-provenance hash, Git SHA, command,
timestamps, host, Python/PyTorch/CUDA versions, visible GPU, metrics, and full epoch
history. The summary command requires five seeds, emits paired effect sizes with
95% bootstrap intervals, and applies BH once over the stated P1-vs-P2 plus
P1-prime-vs-P2 classification family.

## Data-use note

Cx22/CCEDD and other source datasets may restrict use to non-commercial research and
may forbid redistribution. This project publishes derived IDs, coordinates, checksums,
and split assignments—not source images.
