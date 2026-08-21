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

Generated split files contain `crop_id`, `parent_id`, `fold`, `partition`, `protocol`,
and `seed`. Downstream code should join on `crop_id` and must never independently
resplit released crops.

## Data-use note

Cx22/CCEDD and other source datasets may restrict use to non-commercial research and
may forbid redistribution. This project publishes derived IDs, coordinates, checksums,
and split assignments—not source images.

