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

Python 3.11 or newer is required. The lock pins scikit-learn 1.9.0 because grouped
split assignment changed in an older compatible release; accepting an arbitrary
scikit-learn version would make the released P2/P3 hashes machine-dependent.

The clean-clone reproduction entry point is:

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
inside its parent, the dataset total is 4,049, and the five parent counts are
126/108/238/271/223 in release order. Match confidence and the margin over the
runner-up are retained as audit diagnostics, not substituted for the registered
cross-branch agreement criterion. Rows below the diagnostic thresholds remain in
the Branch-B map with `accepted=false`; they are never silently discarded.

`sipakmed_parent_map.parquet` is the canonical filename-derived map used for splits.
`sipakmed_branch_b.parquet` is the independent all-parent image-match audit with
bboxes, confidence, margin, and an `accepted` flag. Keeping them separate ensures
that an ambiguous content match can never overwrite an exact released filename ID.

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
  --output results/classification --max-gpus 3 \
  --architectures resnet18 resnet50 densenet121 efficientnet_b0 mobilenet_v2 vit_b_16 \
  --protocols P1 P1-prime P2 P3
```

Every model JSON records the seed, input-provenance hash, Git SHA, command,
timestamps, host, Python/PyTorch/CUDA versions, visible GPU, metrics, and full epoch
history. The summary command requires five seeds, emits paired effect sizes with
95% bootstrap intervals, and applies BH once over the stated P1-vs-P2 plus
P1-prime-vs-P2 classification family.

On a CUDA-12.8 host that already provides compatible system Torch and torchvision,
create the pinned split/image-library environment without replacing that CUDA build:

```bash
./scripts/setup_cuda128_training_env.sh /path/to/system/python .venv-cu128
```

The setup fails if CUDA 12.8 is unavailable or any split/image dependency differs
from the recorded versions. Model JSONs include all of those versions so an
environment mismatch cannot remain implicit.

### Cx22 segmentation extension

The Cx22 model extension is explicitly a pooled-release stress test; the authors'
parent-disjoint Multi-Train/Multi-Test split is also run as the descriptive
`OFFICIAL` control. First reconstruct the 512-pixel RGB canvases and decode both
released label branches, then attach P3 duplicate components:

```bash
cytofair materialize-cx22 provenance/cx22_parent_map.parquet \
  --labels /path/to/extracted/cx22 --assets /path/to/cx22_assets \
  --output provenance/cx22_model_map.parquet

cytofair audit-duplicates provenance/cx22_model_map.parquet \
  --hashes results/cx22_duplicates/hashes.parquet \
  --pairs results/cx22_duplicates/pairs.csv \
  --output provenance/cx22_model_map_p3.parquet

cytofair cache-cx22 provenance/cx22_model_map_p3.parquet \
  --cache /path/to/cx22_assets/cache256 \
  --output provenance/cx22_cached_model_map_p3.parquet
```

The detached matrix contains five semantic architectures and two Mask R-CNN
instance baselines under P1/P1-prime/P2/P3/OFFICIAL and five fixed seeds (175
cells). Every cell checkpoints after each epoch and the launcher retries the same
cell once before failing closed:

```bash
nohup python scripts/launch_cx22_segmentation_matrix.py \
  --map provenance/cx22_cached_model_map_p3.parquet \
  --output results/cx22_segmentation --max-gpus 3 \
  > results/cx22_segmentation/controller.log 2>&1 < /dev/null &

python scripts/check_cx22_segmentation.py results/cx22_segmentation
```

The status report prints controller health, completed/total cells, GPU assignment,
current epoch, retry count, and exact active/failure log paths. A second invocation
of the launcher skips valid JSON results and resumes any epoch checkpoints.

## Study 2: powered replication and causal decomposition

Study 1 established that crop-level splits share parents and that the parent
fingerprint is learnable, but its six per-architecture tests could not certify the
accuracy inflation. Study 2 is a separate preregistered study
(`PREREGISTRATION_STUDY2.md`, registered as Amendment 6 before any Study 2 model was
trained) with one primary test and four mechanism hypotheses:

| Hypothesis | Design |
|---|---|
| H1 (primary) | One mixed model of SIPaKMeD P1 - P2 accuracy over 12 architectures x 20 new seeds, sized by simulation from the Study 1 pilot |
| H2 | The same model on CRIC (6 architectures x 10 seeds) |
| H3 | 2C factorial: leakage x training diversity at matched crop and parent counts over one common test set |
| H4 | 2B counterfactual background transplants: does a classifier use its test cells' own acquisition context? |
| H5 | 2A gradient-reversal suppression of the parent fingerprint: does the inflation track it? |

H2-H5 form one Holm family. Every split, task, analysis script, and input cache is
pinned by SHA-256 in `reports/study2/registration_hashes.txt`.

The training host limits each user to four CPU cores, so Study 2 decodes crops once
into checksummed caches on node-local storage and augments them on the GPU
(`cytofair.gpu_augment`, an exact port of torchvision's crop sampling). Runs are
deterministic and keep per-crop logits, float16 final weights, and probe-crop features,
so confusion matrices and post hoc analyses never need retraining.

```bash
uv run cytofair study2-design                       # regenerate every registered split
scripts/study2/node_prepare.sh ROOT PYTHON CRIC_IMAGE_DIR   # CRIC + transplant caches
python scripts/study2/launch_matrix.py splits/study2/tasks/*.jsonl \
  --gpus 0,3,4,5 --logs ROOT/runs/logs --status ROOT/runs/status.json
analysis/study2/run_all.sh ROOT PYTHON RSCRIPT     # registered analysis (fails closed)
python scripts/study2/paper_numbers.py            # numbers and tables for paper/v2
```

### Study 2 results (all registered runs complete)

| Test | Estimate | Verdict |
|---|---|---|
| H1 SIPaKMeD P1 - P2 accuracy | +0.97 pp (95% CI 0.59-1.39); all 12 architectures positive | supported |
| H2 CRIC P1 - P2 accuracy | +12.1 pp (error rate 2.3x at slide level) | supported |
| H3 leakage at matched diversity | +1.73 pp; diversity alone 0.00 pp | supported |
| H4 background route | +0.42 pp (-0.35 to 1.18) | not supported |
| H5 gap vs gradient-reversal recoverability | ConvNeXt-T evaded the adversary; slope n.s. | not supported |

### Follow-up: where the leaked signal lives (Amendment 8)

`PREREGISTRATION_STUDY2_AMENDMENT8.md` registered eight further hypotheses (F1-F8, own Holm
family) after the Study 2 analysis and before any follow-up run: CRIC slide-context pull,
test-time carrier ablations and quantitative Grad-CAM on the saved 2C models, carrier removal
during training on fresh seeds, rank-limited LEACE erasure on the saved Phase 1 models, a
stain-normalisation dose, and a probe-matched linear adversary (1,580 tasks). Supported: F2
(test-time stain transfer cuts the leak advantage from 1.72 to 0.46 pp) and F3 (leakage survives
masking every background, +1.86 pp). Not supported: F1 (no slide-level label prior; reversed),
F4 (colour augmentation does not prevent leakage), and F5-F8 (cutting linear field
recoverability by up to 91% leaves the gap unchanged). Grad-CAM maps of leaking and clean models
do not differ. Amendment 9 records a preparer labelling fix made before any F3/F4 estimate was
seen. Outputs: `results/study2/analysis`, `results/study2/followup`; manuscript: `paper/v2`.

```bash
python scripts/study2/build_followup_tasks.py --root ROOT --repository REPO --study2-runs ROOT/runs --python PYTHON
analysis/study2/followup/run_followup.sh ROOT PYTHON RSCRIPT   # registered follow-up analysis
python scripts/study2/figures_followup.py                       # follow-up figures for paper/v2
```

`uv run cytofair risk-report MAP` computes, for any crop-to-parent table, the
closed-form ingredients of crop-level optimism: expected straddle, test crops with a
training sibling, siblings per test crop, the training-diversity ratio, and
class-parent dependence.

## Data-use note

Cx22/CCEDD and other source datasets may restrict use to non-commercial research and
may forbid redistribution. This project publishes derived IDs, coordinates, checksums,
and split assignments—not source images.

The concrete reuse boundary for the ISBI unit, journal extension, F2/F3/F8, and the
fish-domain V2 is recorded in `reports/downstream_reuse.md` without claiming future
experiments as completed work.
