# CytoFair Study 2 preregistration: powered replication and causal decomposition

**Registered:** 2026-10-18 (Asia/Kolkata), in the commit that adds this file.
**Relationship to the original plan:** registered as Amendment 6 of `PREREGISTRATION.md`.
**Status at registration:** no Study 2 model has been trained for any registered cell.
The only Study 2 GPU executions before this commit are one-epoch pipeline smoke tests on
P1 splits (listed in `reports/study2/excluded_attempts.md`); none produced a protocol
contrast, and their metrics were checked only for finiteness.

## 0. Why a second study

Study 1 (frozen 7 September 2026) established structural non-independence, literal pixel
sharing, and a learnable parent fingerprint, but left four limitations that this study
addresses:

1. **Specification.** One question ("does crop-level splitting inflate accuracy?") was tested
   six times, once per architecture, inside a 12-test BH family. Study 2 replaces this with
   one mixed-model test of the protocol effect (H1), powered by simulation.
2. **Availability versus use.** Study 1 showed a fingerprint is available, not that
   classifiers route predictions through it. Study 2 adds a counterfactual background
   transplant (2B, H4) and adversarial fingerprint suppression (2A, H5).
3. **Leakage versus diversity.** At equal crop counts, P1 training sets contain about 914
   distinct parents and P2 training sets about 773, so P1 also has more training
   diversity. Study 2 crosses leakage with diversity at matched crop and parent counts
   over one common test set (2C, H3).
4. **One rescored benchmark.** Study 2 rescores CRIC (H2), whose crop-level split places a
   training parent behind 100% of test cells.

Study 1 results are used only as pilot estimates for the power analysis. No Study 1 run
enters any Study 2 test, and the Study 2 seeds are disjoint from the Study 1 seeds.

While validating the analysis code, the H1 model was fitted to the Study 1 runs
(`results/study2/power/power_pilot.json`): P1 - P2 = 1.37 pp, Satterthwaite p = 0.0054.
That reanalysis was chosen after the Study 1 data were seen. It is reported as a post hoc,
exploratory observation and is not a confirmatory result.

## 1. Fixed training recipe

Every run uses the Study 1 classification recipe: ImageNet initialisation, a new linear
head, AdamW (learning rate 1e-4, weight decay 1e-4), batch size 64, 20 epochs, cross
entropy, final-epoch endpoint, no validation split, no early stopping, no checkpoint
selection. Training augmentation is RandomResizedCrop(224, scale 0.8-1.0, ratio 3/4-4/3),
horizontal flip (p = 0.5), vertical flip (p = 0.5), ImageNet normalisation; evaluation uses
Resize(224 x 224) and the same normalisation.

Execution changes relative to Study 1, none of which alters a model, split, seed, or
endpoint:

* The training host limits each user to four CPU cores (cgroup quota 400%). Crops are
  therefore decoded once into checksummed caches on node-local storage, uploaded to the
  GPU, and augmented there (`cytofair.gpu_augment`). Crop parameters come from an exact
  port of torchvision's `RandomResizedCrop.get_params` (verified identical for 100
  seed/shape pairs, `tests/test_gpu_augment.py`); resizing is antialiased bilinear.
* cuDNN deterministic mode, `torch.use_deterministic_algorithms(True, warn_only=True)`,
  and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
* Every run writes, beside its JSON: per-crop test logits, a float16 copy of the final
  weights, penultimate features for the balanced 424-parent probe crops, and logits for
  any registered extra evaluation set (Section 7).

At most four GPUs (indices 0, 3, 4, 5 on the training host) run concurrently, one job per
GPU.

## 2. Architectures

Twelve ImageNet-initialised backbones (exact weights in `cytofair.backbones.WEIGHT_NAMES`):
the six Study 1 models (ResNet-18, ResNet-50, DenseNet-121, EfficientNet-B0,
MobileNet-V2, ViT-B/16) and six added for breadth (ConvNeXt-T, Swin-T, RegNetY-1.6GF,
EfficientNetV2-S, MaxViT-T, and self-supervised DINOv2 ViT-S/14 at 224 px).

## 3. Seeds and stopping rule

`STUDY2_SEEDS` = 137, 173, 211, 257, 307, 349, 401, 457, 503, 557, 601, 653, 701, 751,
809, 853, 907, 953, 1009, 1051. The first five were named as robustness seeds in the
original preregistration and were never run. Phase 1 uses all twenty; the mechanism arms
and CRIC use the first ten (`MECHANISM_SEEDS`).

No seed, architecture, or cell is added, removed, or replaced after any result is seen.
A failed run (infrastructure failure or non-finite loss) is retried once with the identical
seed and split; a cell that fails twice is reported as missing, and every test that needs it
fails closed rather than substituting a run.

## 4. Splits

All splits were generated before registration and are committed under `splits/study2/`
with SHA-256 hashes in `study2_split_manifest.csv`. Trainers refuse a split whose hash
differs. P1, P1-prime, P2, and P3 follow the original definitions. For all twenty SIPaKMeD
seeds, P2 has exactly 810 test crops, so P1-prime is byte-identical to P1 and is not trained
separately (P1-prime - P2 equals P1 - P2 by construction). For CRIC, P1-prime differs from
P1 by 2-11 training crops and is trained.

## 5. Phase 1: powered SIPaKMeD replication (H1, the primary endpoint)

**Design.** 12 architectures x 20 seeds x {P1, P2, P3} = 720 runs.

**Primary endpoint.** Test accuracy in percentage points.

**Confirmatory model (H1).** On P1 and P2 runs:

    accuracy ~ x + (1 + x | architecture) + (1 | seed) + (1 | split)

with `x = +0.5` (P1) or `-0.5` (P2), so the fixed effect is P1 - P2; `split` is the
seed-by-protocol partition shared by all architectures (omitting it would treat twelve
evaluations of one test set as independent). REML, `bobyqa`, lme4 with lmerTest
Satterthwaite degrees of freedom, two-sided alpha = 0.05.
**Decision rule:** H1 is supported if p < 0.05 and the estimate is positive.

**Convergence.** A singular (boundary) fit is accepted. Only on convergence failure, the
model is simplified in this order: uncorrelated architecture intercept and slope; then
dropping the seed intercept. The architecture slope and the split effect are never dropped.

**Assumptions and fallback.** Conditional residual skewness, excess kurtosis, Shapiro-Wilk,
architecture-slope BLUPs, and residual SD by protocol are reported. If no model converges,
or residual |skewness| > 1, or excess kurtosis > 3, the primary result is the two-way
cluster bootstrap (architectures and seeds resampled independently, 10,000 replicates,
percentile CI and two-sided bootstrap p).

**Sensitivity (reported, not decisive).** Kenward-Roger degrees of freedom; binomial GLMM
on correct counts with an observation-level random effect; seed-level exact sign-flip
test; the cluster bootstrap.

**Power.** `analysis/study2/power_simulation.R` simulated the confirmatory model 1,000
times per design from the Study 1 pilot (effect 1.37 pp; architecture intercept SD 0.44,
slope SD 0.36; split SD 0.31 for P1 and 0.59 for P2; residual SD 0.69; seed SD 0;
intercept-slope correlation set to 0 because six architectures put it on its -1
boundary). The registered rule was: choose the smallest seed count in {10, 15, 20, 25, 30},
with twelve architectures, whose power for a 0.8-pp effect is at least 0.90 under pilot
variances and at least 0.75 with every variance inflated by 1.5. Twenty seeds give 0.977
and 0.782 (fifteen give 0.949 and 0.698). 0.8 pp exceeds the 0.62-pp spread between
published SIPaKMeD scores that motivates the question.

**Secondary analyses (non-confirmatory, no multiplicity claim).** P2 - P3 accuracy (same
model); P1 - P2 for macro-F1 and balanced accuracy; per-architecture paired differences
(descriptive t intervals); Kendall's tau between P1 and P2 architecture rankings with a
seed bootstrap; the P3 <= P2 <= P1 ordering per architecture; pooled confusion matrices,
NLL, and ECE; and two crop-level analyses from the saved predictions:
**M1** within-crop P1 - P2 accuracy for crops that appear in both P1 and P2 test sets
(crop-level bootstrap), and **M2** within-P1 accuracy as a function of the number of
training siblings and literal source-pixel overlap (logistic GLM with crop-clustered
standard errors, adjusting for class and the crop's P2 accuracy).

**Gate A.** If H1 is supported, the inflation effect is reported as certified. If not, the
paper is framed as a careful null replication. The mechanism arms are run regardless
(they are registered now), but their interpretation follows Gate A.

## 6. Arm 2C: leakage x diversity at matched size (H3)

**Design** (`cytofair.factorial`, `FACTORIAL_CONFIG`). For each of the ten mechanism seeds
and each class: 30% of parents with at least two crops are evaluation parents E; each E
parent contributes floor(k/2) test crops (the common test set, about 527 crops) and its
remaining crops are donor siblings. Six parents per class with at least three crops are
reserved as unseen background donors U (never trained on or tested). The rest form the
clean pool B. Four training sets each contain exactly 220 crops per class:

* `clean_high`: every B parent;
* `leak_high`: the E parents' donor siblings replace the last |E| parents of one fixed B
  permutation;
* `clean_low` and `leak_low`: the same construction on the first 70% of that permutation.

Within a diversity level, leak and clean have identical crop counts and identical numbers
of distinct parents (720 versus 504 in total; effective parent numbers 633 and 632 versus
432 and 427), and differ only in whether the test parents' siblings are present. All test
crops have a training sibling in the leak cells (mean 2.0 high, 3.0 low) and none in the
clean cells. Low-diversity parent sets are nested in high-diversity sets. Configurations
with 240 crops per class were infeasible for seed 307 and were rejected before any training.

**Models.** ResNet-50, DenseNet-121, ConvNeXt-T, ViT-B/16; 4 x 10 x 4 = 160 runs.

**H3.** `accuracy ~ leak * div + (1 + leak | architecture) + (1 | seed) + (1 | training_set)`
with leak and div coded +-0.5. H3 is the leak main effect (leakage at matched diversity),
predicted positive. Simplification order on convergence failure: uncorrelated slope, then
intercepts only.

**Secondary.** The diversity effect is estimated only within clean cells
(`clean_high - clean_low`), because the leak dose (siblings per test crop) is higher in the
low-diversity leak cell; the leak x diversity interaction is exploratory for the same
reason. Embedding diversity of every training set is reported in the space of a retrained
RGB parent-probe encoder (`cytofair.fingerprint`). A descriptive decomposition of the
Phase 1 P1 - P2 gap combines the H3 estimate (scaled by the 91.7% leak rate of natural P1)
with the clean diversity effect scaled log-linearly from the 720/504 contrast to the natural
914/773 contrast; it is labelled model-based and approximate.

## 7. Arm 2B: counterfactual background transplant (H4)

Every 2C model is also evaluated on composites of its seed's common test set
(`cytofair.transplant`, plans in `splits/study2/sipakmed_2b/`). The cell foreground is kept
(the Study 1 background-only mask: Otsu on saturation and luminance, 5 x 5 closing) and
feathered (Gaussian sigma 2) onto a donor background inpainted with Telea (radius 5):

* `original`; `self` (own inpainted background: seam and inpainting artifacts only);
* `sibling` (background of a donor sibling of the same parent: own acquisition context);
* `unseen_same_class` (background from a U parent of the same class);
* `unseen_other_class` (one composite per other class, from U parents).

**H4 (context-route difference-in-differences).** For each run,
`route = acc(sibling) - acc(unseen_same_class)` measures how much the recipient's own
acquisition context contributes beyond class-typical context. Leak models have seen that
context with the correct label; clean models have not. H4 is the leak coefficient in
`route ~ leak + div + (1 + leak | architecture) + (1 | seed) + (1 | training_set)`,
predicted positive (fallback: intercepts only).

**Secondary.** The ratio of H4 to the leak effect on original images (the share of the
leakage inflation attributable to the context route); the donor-class probability shift for
other-class backgrounds relative to same-class backgrounds (whether backgrounds carry class
information that classifiers use, independent of leakage); `self` versus `original`
(artifact size); and H4 restricted to recipients whose background covers at least 30% of
the crop.

## 8. Arm 2A: adversarial fingerprint suppression (H5)

A class-conditional parent head (512-unit MLP over the training parents, logits of
other-class parents masked) sits behind a gradient-reversal layer whose coefficient follows
the DANN ramp `lambda * (2 / (1 + exp(-10 p)) - 1)`. The class head is unchanged. Because
the adversary only distinguishes parents within a class, suppressing it never requires
removing class information.

**Design.** ResNet-50 and ConvNeXt-T; lambda in {0, 0.01, 0.03, 0.1, 0.3, 1, 3}; natural P1
and P2 splits of the ten mechanism seeds; 280 runs. Lambda = 0 trains the same head with no
reversed gradient and is the matched baseline. The full sweep, including any degenerate
region, is reported.

**Recoverability R.** Within-class parent identification accuracy of a multinomial logistic
regression (C = 1, standardised features) on the frozen penultimate features of the
balanced 424-parent probe split (two fitting and two evaluation crops per parent; seed 17),
reported with its class-only floor (the accuracy of a representation that knows only the
class). Overall 424-way and nearest-centroid accuracies are secondary.

**Non-degenerate regime.** For each architecture, the lambdas whose mean P2 accuracy is
within 1.0 pp of its lambda = 0 mean. **Manipulation check:** at the largest such lambda,
mean R must fall by at least 25% of its above-floor value at lambda = 0; an architecture that
fails is excluded from H5, and if both fail H5 is reported as untestable, not refuted.

**H5.** Within the regime, per architecture-seed-lambda pair,
`gap = acc(P1) - acc(P2)` and `R_pair` = mean R of the two runs;
`gap ~ architecture + R_pair + (1 | seed)`. H5 is the R_pair slope, predicted positive (the
inflation shrinks as the fingerprint is suppressed).

## 9. CRIC rescore (H2)

CRIC cells are 100 x 100 windows centred on the published nucleus coordinates (shifted
inward at image borders) from the 400 Figshare slide images (MD5-verified,
`data_manifests/cric_images_verified.json`). Six Bethesda classes. Models: ResNet-18,
ResNet-50, DenseNet-121, EfficientNet-B0, ConvNeXt-T, ViT-B/16; ten mechanism seeds;
P1, P1-prime, P2; 180 runs. **H2** is the H1 model applied to CRIC accuracy (P1 - P2),
predicted positive. Secondary: P1-prime - P2, balanced accuracy and macro-F1 (the classes
are imbalanced), and the CRIC versus SIPaKMeD effect sizes against their structural
leakage, which is exploratory.

## 10. Multiplicity

H1 is the single primary endpoint, tested alone at alpha = 0.05. H2, H3, H4, and H5 form
one Holm family (family-wise alpha = 0.05); each is supported only if its Holm-adjusted p
is below 0.05 and its estimate has the predicted sign. Everything else is secondary or
exploratory and carries no confirmatory claim.

## 11. Exclusions and deviations

Runs are excluded only for logged infrastructure failure, non-finite loss, or a failed
integrity check (wrong split hash, cache hash, or missing artifact); never for magnitude or
direction. Every attempt is logged. Any deviation from this document is recorded as a dated
amendment before the affected analysis is run; a deviation discovered afterwards is
reported as such.

## 12. Gate B and venue framing

Gate B is interpretive, not statistical: the causal evidence is "clean" if H3 and H4 are
supported and H5 is either supported or untestable for a stated reason, "murky" otherwise.
It determines framing only. All registered results are reported whatever the gates show.

## 13. Registered artifacts

The analysis code (`analysis/study2/*.R`, `analysis/study2/*.py`), split manifest, task
builder, and caches are fixed by this commit; their SHA-256 hashes are listed in
`reports/study2/registration_hashes.txt`.

## Amendment 7 — failure handling (2026-10-18, Asia/Kolkata)

Registered while Phase 1 was running and before any 2A run had started; no Study 2
result had been analysed or inspected. Two cases were not handled by Section 3:

1. **Infrastructure failures.** A run that fails because of the host rather than the run
   itself (out-of-memory caused by another user's process on the same GPU, loss of the
   host, session, or controller, or a filesystem error, each identifiable from the
   attempt log) does not count toward the one-retry limit. It is rerun with the identical
   seed, split, and code, and every attempt stays in the log.
2. **Divergence in the 2A sweep.** Runs are deterministic, so a run whose training loss
   becomes non-finite diverges again on retry. In the gradient-reversal arm only, such a
   run is recorded as *diverged* rather than missing: a lambda with any diverged run of an
   architecture lies outside that architecture's non-degenerate regime, the diverged runs
   are excluded from H5, and the sweep table reports them with the epoch at which the loss
   became non-finite. A diverged run in any other arm still makes the tests that need it
   fail closed, as registered.

The collector (`analysis/study2/collect_runs.py --diverged-logs`) and `mechanism_lmm.R`
implement point 2; their new hashes are appended to `reports/study2/registration_hashes.txt`.

## Amendment 8 — follow-up experiments (2026-10-19, Asia/Kolkata)

Registered after the Study 2 analysis was complete, before any follow-up model was trained
or any follow-up analysis was run on existing outputs. It adds eight new hypotheses (F1-F8)
on how leakage is used — CRIC slide-context pull, test-time carrier ablations, carrier
removal during training, rank-limited LEACE erasure, a stain-normalisation dose, a
probe-matched linear adversary, and a pooled dose-response — in their own Holm family. It
does not change H1-H5 or their verdicts. Full text: `PREREGISTRATION_STUDY2_AMENDMENT8.md`.

## Amendment 9 — preparer bug fix (2026-10-20, Asia/Kolkata)

A labelling bug in `analysis/study2/followup/prepare_followup.py` (standard-recipe carrier runs
labelled `colour`) was fixed before any F3 or F4 estimate was seen. Details in
`PREREGISTRATION_STUDY2_AMENDMENT8.md`.
