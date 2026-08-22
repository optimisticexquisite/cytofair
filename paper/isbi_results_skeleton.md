# Do We Train on Test Cells? Provenance Leakage in Public Cervical Cytology Benchmarks

> Results-first ISBI skeleton. Text in square brackets remains blocked on human
> verification and must not appear in a submission.

## Abstract

Public cervical-cytology benchmarks often release several cell crops from one source
field, but evaluation commonly treats those crops as independent. We recover exact
crop-to-parent provenance for SIPaKMeD and Cx22, quantify train/test parent sharing,
and release fixed parent-disjoint splits. Two independent SIPaKMeD recovery branches
agree on 4,049/4,049 crops, and all recovered boxes lie inside their source images.
Across five fixed 80:20 crop-level splits, 45.63% ± 1.04% of SIPaKMeD parents and
34.01% ± 1.10% of Cx22 parents straddle train and test; 92.62% ± 0.40% and
85.23% ± 3.51% of their test crops, respectively, have a parent in training. The
one-cell-per-image Herlev control is exactly 0.00% ± 0.00% on both endpoints.
Moreover, 52.62% ± 1.97% of SIPaKMeD test crops share literal pixels with a training
crop. A parent-identity probe reaches 25.73% ± 0.66% accuracy against 0.24% chance;
the shuffled-label control is 0.17% ± 0.24% and its 95% CI covers chance. Across six
architectures, P1 accuracy ranges from 97.14% ± 0.95% to 98.12% ± 0.28%, whereas P2
ranges from 95.16% ± 1.50% to 96.79% ± 0.42%. Every P1−P2 point estimate is positive,
but none of the 12 registered P1/P2 and P1′/P2 tests survives BH correction. CytoFair
provides provenance maps, fixed splits, and a raw-to-table reproduction command.

## 1. Introduction

The unit supplied to a classifier is not necessarily the independent unit sampled
from a population. In cervical cytology, multiple released cell crops can inherit
stain, illumination, focus, background texture, and even pixels from the same source
field. A random crop-level split can therefore place observations from one source on
both sides of evaluation. We measure this non-independence as a property of the
benchmarks and provide evaluation assets that preserve parent independence.

Our contributions are: (1) exact, independently checked provenance recovery;
(2) analytic and observed structural-leakage measurements with a negative-control
dataset; (3) a direct parent-identity probe and literal pixel-overlap audit; and
(4) parent-disjoint P2 and near-duplicate-component-disjoint P3 splits, evaluated
against both ordinary P1 and the P1′ size-matched random control.

## 2. Data and provenance

SIPaKMeD contains 4,049 crops from 966 released cluster fields. Filename recovery
(Branch A) and exhaustive same-class image matching (Branch B) agree on all 4,049
crops; all 4,049 recovered boxes are contained. The reconstructed class-parent
counts are 126 Superficial-Intermediate, 108 Parabasal, 238 Koilocytotic,
271 Metaplastic, and 223 Dyskeratotic.

Cx22's three author releases contain 1,320 generator ROIs drawn from 514 of the 686
CCEDD source images. All source names resolve and all edge-clipped boxes are inside
2048×1536 parents. Multi-Train and Multi-Test share no parent, but Pair shares 233
parents with Multi-Train and 55 with Multi-Test. CRIC supplies 11,534 classified
cells from 400 source images. Herlev supplies 917 cells with one parent per cell and
is the negative control.

## 3. Leakage measurements and mechanism

For a parent with *k* crops under independent 80:20 assignment, the probability of
remaining wholly within one arm is `0.8^k + 0.2^k`; hence the straddle probability is
`1 - 0.8^k - 0.2^k`. The observed five-seed means lie inside 10,000-allocation 95%
Monte Carlo intervals for every dataset. Herlev remains exactly zero, while CRIC's
larger parent clusters yield 94.35% ± 1.10% straddled parents and 99.97% ± 0.08%
leaked test cells.

SIPaKMeD contains 2,185 overlapping within-parent crop pairs. Under P1,
52.62% ± 1.97% of test crops share pixels with training; P2 and P3 are both
0.00% ± 0.00%. The parent probe reaches 25.73% ± 0.66% in RGB, compared with
0.24% chance. Accuracy remains 13.80% ± 1.12% in grayscale, 17.31% ± 0.64% after
stain normalization, 18.68% ± 1.25% with only background, and 16.27% ± 0.60% after
high-pass filtering. The RGB label-shuffled control is 0.17% ± 0.24%; its five-seed
95% CI of −0.14% to 0.47% covers the 0.24% chance baseline.

Thus SIPaKMeD crops carry a parent-field fingerprint at more than two orders of
magnitude above chance. The 18.68% ± 1.25% background-only result shows that the
signal is not confined to
cell morphology or label content, while its persistence across all four transforms
does not identify a single exclusive substrate. Crop-level splitting can expose this
source-specific signal to both training and evaluation; the one-cell-per-parent
Herlev control has no corresponding parent-sharing structure. This measures the
confound directly rather than inferring it only from a downstream accuracy change.

## 4. Matched re-scoring

The main comparison contains all four protocols. P1 is a stratified crop split. P2
keeps every recovered parent in one arm. P3 additionally keeps each thresholded
near-duplicate parent component in one arm. P1′ retains P1's test crops and, when
needed, downsamples only its training arm to the exact P2 training count. On
SIPaKMeD every fixed-seed protocol contains 3,239 training and 810 test crops, so
P1′ is identical to P1: the ordinary random split already has exactly the P2 data
volume. Thus the P1′−P2 contrast isolates split unit from training-set size and is a
main-text result, not a supplementary analysis.

The matched accuracy results are:

| Architecture | P1 | P1′ | P2 | P3 |
|---|---:|---:|---:|---:|
| DenseNet-121 | 0.9812 ± 0.0028 | 0.9812 ± 0.0028 | 0.9672 ± 0.0069 | 0.9738 ± 0.0050 |
| EfficientNet-B0 | 0.9778 ± 0.0049 | 0.9778 ± 0.0049 | 0.9669 ± 0.0149 | 0.9741 ± 0.0071 |
| MobileNet-V2 | 0.9741 ± 0.0045 | 0.9741 ± 0.0045 | 0.9516 ± 0.0150 | 0.9620 ± 0.0093 |
| ResNet-18 | 0.9719 ± 0.0059 | 0.9719 ± 0.0059 | 0.9600 ± 0.0089 | 0.9635 ± 0.0128 |
| ResNet-50 | 0.9751 ± 0.0047 | 0.9751 ± 0.0047 | 0.9679 ± 0.0042 | 0.9627 ± 0.0053 |
| ViT-B/16 | 0.9714 ± 0.0095 | 0.9714 ± 0.0095 | 0.9556 ± 0.0084 | 0.9553 ± 0.0168 |

Paired P1−P2 differences range from 0.0072 ± 0.0083 to 0.0225 ± 0.0163 accuracy.
The identical P1′−P2 differences confirm that these estimates are not explained by a
training-count mismatch in SIPaKMeD. All 12 primary tests use one declared BH family;
the smallest adjusted value is q=0.1109, so none meets the 0.05 threshold. Table 2b
reports every raw p and q with paired Cohen's dz and a 10,000-resample 95% CI.

The registered P3 ≤ P2 ≤ P1 accuracy ordering passes for ResNet-50 and ViT-B/16 but
fails for DenseNet-121, EfficientNet-B0, MobileNet-V2, and ResNet-18 because their P3
means exceed P2. These violations are retained rather than repaired. P3 uses a
different grouped test fold, so a more conservative structural grouping need not
produce monotonically lower accuracy in a five-seed finite sample. P1-to-P2
architecture rankings have Kendall τ=0.467 with a seed-bootstrap 95% CI of
0.071–0.733; 4/15 architecture pairs reverse and no ties change.

## 5. Scope, self-audit, and release

This is a resource and measurement study. Affected prior results are discussed in
aggregate; no individual paper is labelled deficient. The planning corpus reports
six lab SIPaKMeD studies and three lab Cx22 studies. The records currently mapped
with explicit dataset evidence are retained below; the self-audit gate remains open
until the remaining title-to-dataset mappings are checked against the papers.

| Benchmark | Lab papers included in protocol audit |
|---|---|
| SIPaKMeD | GC-EnC; O-bHSA; GWABHO-Adam; RWAEFA; Hybrid QCNN ensemble; CrCytoGan [verify each final record] |
| Cx22 | CerviSegNet; ReSEU-Net++; [third corpus record pending verification] |

Grouped scores are estimates at a more defensible deployment unit; a lower apparent
score does not imply lower clinical utility. CytoFair releases derived provenance,
fixed split hashes, near-duplicate sensitivity groups, raw seed-level results, and a
single raw-download-to-Table-1 command without redistributing restricted images.

## Submission gates still open

- Record a successful clean-clone run on a host that has never seen the project.
- Ask a colleague who has read only Fig. 1 to state the finding; record their wording verbatim.
- Verify the nine-paper dataset crosswalk against final bibliographic records; do not mark the self-audit gate passed before then.
