# Study 2, Amendment 8 -- Follow-up experiments on how leakage is used

Registered 2026-10-19 (Asia/Kolkata) as an amendment to `PREREGISTRATION_STUDY2.md`.

## 0. Status, motivation, and what was known

This amendment was written **after** the registered Study 2 analysis was complete. Its
verdicts stand and are not revisited here: H1 (SIPaKMeD P1 - P2) and the Holm-family
hypotheses H2 (CRIC) and H3 (leakage at matched diversity) were supported; H4 (context route
through backgrounds) and H5 (gap tracks gradient-reversal suppression) were not.

The eight hypotheses below (F1-F8) are new questions, motivated by two observations from
Study 2: (i) leak-trained 2C models kept most of their advantage on backgrounds transplanted
from unseen fields, which points to a carrier inside the cell region rather than the
background; and (ii) for ConvNeXt-T the gradient-reversal adversary was evaded (independent
probe recoverability rose), so H5 had only one usable architecture.

**Known when writing:** every registered Study 2 output, including transplant condition
means by cell and the gradient-reversal sweep means. **Not known:** any per-cell CRIC result
broken down by slide composition; any evaluation of ablated images; any evaluation of erased
features; any Grad-CAM map; any model trained with masked backgrounds, colour augmentation,
stain normalisation, or the linear adversary. One-epoch smoke tests used only crop-level or
leak-cell runs, so no follow-up contrast could be observed before registration.

Results for F1-F8 are reported as registered follow-ups to Study 2, separately from H1-H5,
whatever they show.

## 1. Common elements

The Study 2 recipe (section 1 of the parent document) is unchanged: ImageNet initialisation,
AdamW 1e-4 / weight decay 1e-4, batch 64, 20 epochs, final-epoch endpoint, random resized crop
(0.8-1.0) and flips on the GPU, deterministic kernels. Only the factor each arm manipulates
differs. Every new training run saves per-crop logits, final weights, and probe-split
features. Failure handling follows Amendment 7: infrastructure failures are rerun; a
linear-adversary run whose attempts both end in a non-finite loss is recorded as *diverged*,
its lambda leaves that architecture's regime, and it is excluded from F7 and F8; any other
failure fails the arm closed.

**Recoverability R** (F5-F8) is within-class parent identification accuracy on the balanced
424-parent probe split (seed 17; two fitting and two evaluation crops per parent): multinomial
logistic regression on standardised features with sklearn's objective (C = 1), solved by
L-BFGS on the GPU (`cytofair.followup_eval.fit_logistic`), restricted at prediction to parents
of the crop's class. The same function scores every arm. Its agreement with the registered
H5 sklearn probe on the 280 Study 2 gradient-reversal runs is reported (secondary).

## 2. F1 -- CRIC slide-context pull (existing predictions; no training)

For every CRIC cell, the **context class** is the most frequent class among the *other*
cells of its slide image; cells whose slide has no other cell, or a tied majority, are
excluded. A cell is **discordant** if its own class differs from its context class. For each
of the 120 registered CRIC P1 and P2 runs (6 architectures x 10 seeds), the softmax
probability of the context class is computed for every test cell. Per cell and protocol,
probabilities are averaged over all runs in which the cell is a test cell.

**F1** is the mean, over discordant cells tested under both protocols, of
`100 * (p_context(P1) - p_context(P2))`, predicted positive: when a cell's slide is in
training, predictions are pulled toward the slide's majority label. Inference: percentile
cluster bootstrap over slides (10,000 replicates, seed 20261019), two-sided
`p = 2 * min(P(boot <= 0), P(boot >= 0))`.
Secondary: the within-cell P1 - P2 accuracy difference in concordant and discordant cells
and their contrast (predicted concordant > discordant); the rate of predicting the context
class; per-architecture F1.

## 3. F2 -- test-time carrier ablations (existing 2C checkpoints; no training)

The 160 registered 2C models are re-evaluated on their seed's common test set under six
conditions, each rendered at native resolution (`cytofair.carriers`): `original`; `grey`
(luminance only); `stain_transfer` (Reinhard transfer of the crop's LAB mean and standard
deviation to those of the 2B unseen same-class donor crop); `low_pass` (Gaussian blur, sigma =
2 px at the 224-px input); `illumination_flat` (subtract a sigma = 16 px Gaussian, add back the
crop mean); `background_grey` (pixels outside the Study 1 foreground mask set to the ImageNet
mean). Ablation plans are `splits/study2/followup/ablations/ablation_plan_seed*.csv`.

For each ablation `a`, with run-level accuracy in pp and `ablated = +-1/2`:
`acc ~ leak * ablated + div + (1 + leak | architecture) + (1 | seed) + (1 | training set) + (1 | run)`
(fallback: random intercepts only). The reduction of the leak effect is `-coef(leak:ablated)`.
**F2** is that reduction for `stain_transfer`, predicted positive (the leak advantage depends
on the crop's own colour statistics). Secondary: the reductions for the other four ablations;
the leak effect within each condition; agreement of re-evaluated originals with the recorded
predictions (fp16 weights).

**Grad-CAM (descriptive only).** For ResNet-50, DenseNet-121, and ConvNeXt-T 2C models, the
Grad-CAM map of the true-class logit at the last convolutional stage is upsampled to the crop
and the fraction of positive attribution outside the foreground mask is averaged per run. The
leak effect on that fraction is estimated with the H4 intercept model and reported with its
CI and no test. A sanity check compares each map with the map obtained after replacing the
classification head by a fixed random head (median Spearman correlation).

## 4. F3 and F4 -- carrier removal during training (new runs)

Ten seeds never used for any model, each checked feasible for the registered 2C
configuration: 1103, 1151, 1201, 1249, 1301, 1361, 1409, 1451, 1499, 1553. Their 2C designs
and cell splits are `splits/study2/followup/sipakmed_2c/` (manifest
`splits/study2/followup/followup_split_manifest.csv`). ResNet-50, DenseNet-121, ConvNeXt-T,
and ViT-B/16 are trained on all four cells under three conditions: `standard` (Study 2
recipe), `masked` (every training and test crop has its background set to the ImageNet mean;
cache `sipakmed_masked`), and `colour` (training batches additionally receive HED-strong
optical-density jitter, alpha in [0.8, 1.2] and beta in [-0.2, 0.2] per stain channel,
followed by brightness and contrast factors in [0.8, 1.2]; test crops unmodified). 480 runs.

**F3**: within `masked`, the H3 model
`acc ~ leak * div + (1 + leak | architecture) + (1 | seed) + (1 | training set)` (H3 fallback
ladder); the leak coefficient, predicted positive (leakage survives without any background).
**F4**: within `standard` and `colour`, with `treated = +-1/2`,
`acc ~ leak * treated + div + (1 + leak | architecture) + (1 | seed) + (1 | training set)`
(fallback: random intercepts); `-coef(leak:treated)`, predicted positive (colour augmentation
shrinks the leak effect). Secondary: the H3 model on `standard` (an independent replication
of H3 on fresh seeds); the same contrast for `masked` versus `standard` (background share of
the leak effect); cell means.

## 5. F5 -- LEACE erasure of within-class parent identity (existing Phase 1 checkpoints)

For every registered Phase 1 P1 and P2 model (12 architectures x 20 seeds; 480 models),
penultimate features of all SIPaKMeD crops are extracted (eval transform). On the model's own
training crops, with `W = Sigma^{-1/2}` (ridge 1e-6 x mean eigenvalue) and the within-class
between-parent scatter `S = sum_j (n_j / n)(mu_j - mu_class(j))(mu_j - mu_class(j))^T`, the
rank-r eraser removes the top-r eigendirections `U_r` of `W S W`:
`x' = x - (x - mu) W U_r U_r^T W^{-1}` (rank-limited LEACE; Belrose et al., 2023). Ranks
r in {0, 4, 16, 64, 256}. At each rank the linear class head is refitted on the erased
training features (the logistic objective above) and test accuracy is measured; R is
measured on the erased probe features.

Per architecture, the **regime** is the set of ranks whose mean P2 accuracy is within 1 pp of
rank 0 (refitted heads); r* is its largest rank. **Manipulation check**: mean R_pair at r* is at
least 25% of the above-floor rank-0 value below it (as for H5); architectures that fail, or
whose regime is {0}, are excluded. For each included architecture-seed,
`reduction = gap(0) - gap(r*)` with `gap = acc(P1) - acc(P2)`. **F5**: intercept of
`reduction ~ 1 + (1 | architecture) + (1 | seed)` (fallback `(1 | architecture)`), predicted
positive (erasing linearly decodable within-class field identity closes part of the gap).
If no architecture is included, F5 is untestable, not refuted.

## 6. F6 -- stain-normalisation dose (new runs)

Reinhard normalisation of every crop toward one reference (the mean over all SIPaKMeD crops of
per-crop LAB means and of per-crop LAB standard deviations), blended in LAB space at strength
s in {0.5, 1.0} and applied to training, test, and probe crops alike (caches
`sipakmed_reinhard50`, `sipakmed_reinhard100`). ResNet-50, ConvNeXt-T, DenseNet-121, and
EfficientNet-B0 on natural P1 and P2 for the ten Study 2 seeds no mechanism arm used (601,
653, 701, 751, 809, 853, 907, 953, 1009, 1051): 160 runs. Strength 0 is the corresponding
registered Phase 1 run (identical deterministic recipe, split, and seed). **F6**: in
`gap ~ architecture + s + (1 | seed)` over the three strengths, `-coef(s)`, predicted positive.
The regime and manipulation check of section 5 decide which strengths enter F8.

## 7. F7 -- probe-matched linear adversary (new runs)

The 2A design with a different adversary: per-feature standardisation (BatchNorm without
affine parameters) followed by one linear layer over the training parents, class-conditional
masking, gradient reversal with the DANN ramp; the adversary has its own AdamW optimizer
(learning rate 1e-3, no weight decay) and is re-initialised, with fresh optimizer state, every
5 epochs from a forked random stream. ResNet-50, ConvNeXt-T, and DenseNet-121; lambda in
{0, 0.1, 0.3, 1, 3}; natural P1 and P2; seeds as in F6; 300 runs. Regime, manipulation check,
and model exactly as H5: `gap ~ architecture + R_pair + (1 | seed)` over included
architectures and in-regime lambdas. **F7** is the R_pair slope, predicted positive.

## 8. F8 -- pooled dose-response

All included architecture x level cells of F5 (ranks), F6 (strengths), and F7 (lambdas),
after their regimes and manipulation checks:
`gap ~ cell + R_pair + (1 | manipulation:seed)` with one fixed intercept per
manipulation x architecture cell. **F8** is the R_pair slope, predicted positive (across
independent ways of removing the fingerprint, the inflation tracks how much of it remains).

## 9. Multiplicity and interpretation

F1-F8 form one Holm family (family-wise alpha = 0.05), separate from H1-H5. Each is supported
only if its Holm-adjusted p < 0.05 and its estimate is positive; estimates are oriented so the
predicted direction is positive. Everything else here is secondary or descriptive.

Pre-specified reading: F2 and F3 supported together locate the leaked signal in the cell's own
appearance rather than its background; F4 supported identifies colour augmentation as a
partial mitigation; F5, F7, or F8 supported means the inflation tracks recoverable field
identity (the shortcut account); F5 not supported while R falls means the inflation is not
carried by linearly decodable within-class field identity in pooled features.

## 10. Analysis order and artifacts

Each arm is analysed once all of its registered tasks are complete (`--allow-incomplete`
outputs are progress reports only); the Holm family is computed once all eight are available.
The code (`src/cytofair/carriers.py`, `src/cytofair/followup_eval.py`, the trainer options,
`scripts/study2/build_followup_tasks.py`, `analysis/study2/followup/*`), split and ablation
manifests, and task files are fixed by this commit; their SHA-256 hashes are listed under
"Amendment 8" in `reports/study2/registration_hashes.txt`, together with the hashes of the
derived caches. A synthetic dry run of `followup_lmm.R`
(`analysis/study2/followup/simulate_followup_inputs.py`) recovered every planted effect before
registration.

## Amendment 9 -- preparer bug fix (2026-10-20, Asia/Kolkata)

When the registered models were first run on the completed carrier arm, `followup_lmm.R`
stopped at F4 because `prepare_followup.py` had labelled all 160 standard-recipe runs as
`colour`: the label was derived from a DataFrame row in which a missing colour augmentation
(`None` in the run record) had become a truthy NaN. The masked runs were labelled correctly.
The script halted before writing any output, so no F3 or F4 estimate was seen. The fix
derives each run's condition from its own record (`carrier_condition`: background-masked
cache -> `masked`; `color_augment == "hed_strong_bc"` -> `colour`; otherwise `standard`), which
is the registered definition; no model, rule, or data changes. A unit test
(`tests/test_followup_prepare.py`) covers it. The synthetic dry run had not exercised the
preparer, which is why the error was not caught before registration.

Known when this amendment was written: the arm-level results of F1, F2 (with the Grad-CAM
description), and F5, analysed with the registered code once their arms were complete, as
section 10 allows. Not known: any F3, F4, F6, F7, or F8 estimate.
