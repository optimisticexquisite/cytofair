# CytoFair statistical analysis preregistration

**Registered:** 2026-08-22 (Asia/Kolkata)  
**Status at registration:** no dataset has been downloaded into this repository, no model has been trained, and no result has been computed or committed.  
**Target analysis:** provenance leakage in public cervical-cytology benchmarks.

This document fixes the confirmatory analyses before result generation. Any later
change will be appended in an explicitly dated amendment; the original text will
remain recoverable in Git. Exploratory analyses will be labelled as exploratory.

## Data units and datasets

- SIPaKMeD: crop is the observation; cluster-field image is the parent unit.
- Cx22: generated ROI is the observation; source CCEDD image is the parent unit.
- CRIC: classified cell is the observation; source smear image is the parent unit,
  upgraded to patient when a reliable patient identifier is available.
- Herlev: released single-cell image is both observation and parent. It is the
  negative-control arm and must have no multi-observation parent structure.

No image-derived cluster may replace a released or exactly recovered parent ID in
the confirmatory analysis. Proxy parents, if needed, are a separately labelled
lower-bound sensitivity analysis.

## Provenance acceptance criteria

SIPaKMeD and Cx22 provenance is accepted only when all applicable checks pass:

1. Metadata/filename recovery (Branch A) and independent image matching (Branch B)
   agree for at least 99% of observations mapped by both branches.
2. Every recovered bounding box is non-negative and contained in its parent image.
3. Exactly one parent is assigned to every accepted observation; per-parent crop
   counts sum to the released dataset total.
4. SIPaKMeD parent counts by published class equal 126, 108, 238, 271, and 223 in
   the paper's class order (Superficial-Intermediate, Parabasal, Koilocytotic,
   Dyskeratotic, Metaplastic), and crop counts equal the official release totals.
5. Match failures, ambiguous matches, duplicate IDs, and any exclusion are retained
   in an audit table rather than silently dropped.

## Split protocols

Each train/test allocation is generated independently for each seed, stratified by
class whenever mathematically possible, with test fraction 0.20.

- P1: random crop-level split.
- P1-prime: random crop-level split with training observations subsampled to the
  exact P2 training size, class-balanced as closely as integer counts permit.
- P2: parent-disjoint split.
- P3: parent-disjoint split after near-duplicate components are purged or kept
  wholly within one partition.

Seeds are `17, 29, 43, 71, 101` for all confirmatory comparisons. Headline analyses
may later add five prespecified robustness seeds (`137, 173, 211, 257, 307`), but
the original five may not be selectively replaced.

## Structural leakage endpoints

For P1, report (i) the fraction of parents that straddle train and test and (ii) the
fraction of test crops whose parent occurs in training. For a parent with `k` crops,
the analytic probability of straddling is

`1 - (0.8^k + 0.2^k)`.

The expected dataset straddle fraction is the arithmetic mean of this probability
over eligible parents. The observed mean and SD over the five seeds must agree with
the analytic expectation within a two-sided 95% Monte Carlo interval computed from
10,000 allocations using the same parent crop-count vector and allocation rule.
Herlev must have a straddle fraction and leaked-test-crop fraction of exactly zero.

## Mechanistic parent-identity probe

Parents with at least four crops are eligible. Within every eligible parent, crops
are divided 50:50 between train and test without moving an entire parent into one
partition. A ResNet-18 parent classifier is evaluated under RGB, grayscale,
stain-normalised, background-only, and high-pass conditions. Chance is the majority
class accuracy and, when the test set is balanced by construction, `1 / n_parents`.
A label-shuffled run using the same images, optimiser, epochs, and seeds must have a
95% confidence interval covering the chance baseline; otherwise the probe is not
accepted.

## Model-comparison endpoints

The primary endpoint is paired seed-level `P1 - P2` test accuracy for classification
and Dice for segmentation. The key secondary endpoint is `P1-prime - P2`, which
controls training-set size. `P2 - P3` is secondary. The expected leakage-severity
ordering is `P3 <= P2 <= P1` for each architecture's across-seed mean; violations
will be reported, not repaired by seed selection.

Each protocol/architecture/dataset cell is reported as mean plus or minus sample SD
over at least five seeds. A p-value is never reported without an effect size and 95%
confidence interval.

## Tests, multiplicity, and effect sizes

For each paired P1-vs-P2 or P1-prime-vs-P2 contrast, normality of paired differences
is assessed with Shapiro-Wilk at alpha 0.05. A paired t-test is used when normality is
not rejected; otherwise a two-sided Wilcoxon signed-rank test is used. Zero-only
differences are reported descriptively without a fabricated p-value.

Effect size is paired-samples Cohen's dz for t-tests and matched rank-biserial
correlation for Wilcoxon tests, each with a 10,000-resample paired bootstrap 95% CI.
The confirmatory multiplicity family is all P1-vs-P2 and P1-prime-vs-P2 tests in the
dataset-by-architecture grid within a task (classification or segmentation).
Benjamini-Hochberg correction at false-discovery rate 0.05 is applied separately to
those two task families. Both raw p and BH-adjusted q are retained.

Across architectures, use a Friedman test followed only after rejection by Nemenyi
post-hoc comparisons. Ranking change between P1 and P2 is Kendall's tau with a
10,000-resample seed-level bootstrap CI.

## Exclusions and missing runs

Runs may be excluded only for a logged infrastructure failure, non-finite loss, or a
predefined data-integrity failure. The seed is rerun unchanged. All attempts remain
in the run ledger. No result is excluded for magnitude or direction. If fewer than
five valid paired seeds remain, the cell is incomplete and no confirmatory p-value
is emitted.

## Reproducibility gate

The release is accepted only after a clean clone on a machine with no project cache
recreates Table 1 from declared raw-download URLs and checksums through one command.
The run records environment versions, input checksums, seed, command, timestamps,
host, and Git SHA. The verification report will use `git log --follow` to demonstrate
that this preregistration commit predates the first result commit.

## Amendment 1 — factual class-order correction (2026-08-22, Asia/Kolkata)

Before any SIPaKMeD result was computed, the Branch-A integrity gate exposed that
criterion 4 above transcribed the final two class names in the wrong order. The
published/release vector `126, 108, 238, 271, 223` follows the website/archive order:
Superficial-Intermediate, Parabasal, Koilocytotic, **Metaplastic**, Dyskeratotic.
Accordingly the fixed mapping is Metaplastic = 271 parents and Dyskeratotic = 223
parents. No numeric target changed and no observed effect informed this correction.
The original preregistration remains recoverable as Git commit `67e8ded`.

## Amendment 2 — Cx22 model-rescoring extension (2026-08-24, Asia/Kolkata)

This amendment is registered after the completed structural audit of Cx22 but
before any Cx22 segmentation model is trained or any Cx22 model result is
computed. The already known parent counts and structural leakage rates motivate
the experiment but cannot determine a model-score contrast. The earlier
SIPaKMeD results and the original text above remain unchanged.

### Estimand and datasets

The confirmatory estimand is the effect of evaluation unit on segmentation when
all 1,320 released Cx22 ROIs (Pair, Multi-Train, and Multi-Test) are pooled and
repartitioned. This is a stress test of a documented downstream reuse pattern,
not an audit failure assigned to the Cx22 authors. Their released Multi-Train to
Multi-Test partition is parent-disjoint and is retained as a descriptive control
(`OFFICIAL`), with Pair excluded exactly as in the release. It is never pooled
with the confirmatory contrasts.

RGB ROIs are reconstructed from the named CCEDD sources by the released MATLAB
generator rule: crop the requested source rectangle, clip at the source boundary,
resize its longest requested dimension to at most 512 pixels, and paste at the
top-left of a white 512 by 512 canvas. Released cytoplasm and nucleus instance
masks must both resolve for every ROI. Materialization is rejected if any ROI or
mask count differs from 1,320, if any output is not 512 by 512, or if any ROI has
an empty cytoplasm or nucleus target.

### Tasks, models, and fixed training recipe

The semantic task predicts two non-exclusive binary channels: union of all
cytoplasm instances and union of all nucleus instances. Its primary endpoint is
the arithmetic mean of per-image cytoplasm Dice and per-image nucleus Dice.
Channel-specific Dice and IoU are secondary. The fixed model grid is:

- U-Net with ResNet-34 encoder;
- U-Net++ with ResNet-34 encoder;
- DeepLabV3+ with ResNet-50 encoder;
- FPN with ResNet-34 encoder; and
- MAnet with ResNet-34 encoder.

The instance task predicts cytoplasm instances. Its primary endpoint is
per-image mean ground-truth-instance Dice after maximum-weight bipartite matching,
with unmatched ground-truth instances assigned Dice zero. Detection recall at
matched Dice at least 0.70 and matched IoU are secondary. The fixed model grid is
Mask R-CNN ResNet-50-FPN and Mask R-CNN ResNet-50-FPN-v2.

All encoders use ImageNet/COCO initialization where available. Images and masks
are resized to 256 by 256. Training uses AdamW, learning rate `1e-4`, weight decay
`1e-4`, 30 epochs, no early stopping or validation-based checkpoint selection,
and final-epoch evaluation. Semantic batch size is 16; instance batch size is 2.
Aligned random horizontal/vertical flips and 90-degree rotations are the only
augmentations. The five confirmatory seeds remain 17, 29, 43, 71, and 101.

### Protocols and inference

P1, P1-prime, P2, and P3 retain their definitions above. P3 uses the already
registered pHash-and-dHash consensus thresholds of six and groups every connected
parent component wholly within one partition. Every run saves the exact split
hash, input-manifest hash, runtime, Git SHA, epoch history, and final predictions'
aggregate metrics. Interrupted cells resume from an epoch checkpoint without
changing their seed or allocation; failed attempts remain in the ledger.

For each semantic architecture the paired primary contrasts are P1 minus P2 and
P1-prime minus P2. Their ten p-values form one Cx22 semantic Benjamini-Hochberg
family. The corresponding four instance-task contrasts form a separate Cx22
instance family. The test-selection, effect-size, bootstrap, completeness, and
ordering rules in the original preregistration apply unchanged. `OFFICIAL` is
reported as mean plus or minus sample SD over five seeds but is descriptive and
outside both multiplicity families because it excludes Pair and targets a
different released sample.

## Amendment 3 — numerical execution detail (2026-08-24, Asia/Kolkata)

Before any Cx22 epoch completed or emitted a metric, server preflight established
that deterministic full-FP32 execution left the three A6000s predominantly idle
or made a single U-Net epoch exceed 4.5 minutes. The fixed recipe therefore uses
CUDA FP16 autocast with dynamic gradient scaling for both segmentation tasks.
Losses and reported metrics remain accumulated in FP32. This changes no model,
split, seed, batch size, optimizer, epoch count, checkpoint-selection rule, or
statistical endpoint. The reason for the change is recorded throughput, not an
observed segmentation score; no pre-amendment Cx22 checkpoint or result exists.
