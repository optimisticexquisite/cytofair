# Phase 1 provenance recovery report

**Analysis date:** 2026-08-22  
**Registered acceptance contract:** `PREREGISTRATION.md`  
**Status:** SIPaKMeD and Cx22 provenance gates pass; no observations were excluded.

## Deterministic recovery results

| Dataset | Released observations | Recovered parents used | Recovery evidence | Geometry result |
|---|---:|---:|---|---|
| SIPaKMeD | 4,049 | 966 | Filename Branch A and exhaustive same-class image-matching Branch B agree on 4,049/4,049 common crops (100.00%) | 4,049/4,049 Branch-B boxes are inside their parent |
| Cx22 | 1,320 | 514 | Source names and requested ROIs decoded from the authors' three MATLAB generator releases; all sources resolve in the 686-image CCEDD release | 1,320/1,320 clipped generator boxes are inside their 2048×1536 parent |
| CRIC | 11,534 | 400 | Released classifications table supplies source-image identity and cell coordinates | Coordinates recovered; no reliable patient identifier is present in the released table |
| Herlev | 917 | 917 | One released cell image is one parent unit | Negative-control invariant: exactly one observation per parent |

The SIPaKMeD class counts independently reconstructed from the release are:

| Class | Crops | Parents |
|---|---:|---:|
| Superficial-Intermediate | 831 | 126 |
| Parabasal | 787 | 108 |
| Koilocytotic | 825 | 238 |
| Metaplastic | 793 | 271 |
| Dyskeratotic | 813 | 223 |
| **Total** | **4,049** | **966** |

This ordering corrects the class-name swap in the initial planning note; the numeric
vector itself was unchanged and the correction was committed before any result.

## SIPaKMeD Branch-B audit

Branch B ranked every same-class parent at quarter resolution, refined the five best
candidates at full resolution, and never used Branch A's parent assignment in a crop
query. Its minimum winning normalized cross-correlation was 0.999969. The auxiliary
runner-up-margin diagnostic flags 3,195/4,049 matches at the conservative 0.02
threshold. The other 854 rows are retained, not excluded: every one has a contained
box and the independently recovered parent still agrees with Branch A. This diagnostic
is therefore not substituted for the registered 99% cross-branch agreement gate.

The recovered boxes yield 13,064 within-parent crop pairs. Of these, 2,185 share at
least one literal pixel and the maximum pairwise IoU is 0.546. Across the five fixed
P1 seeds, 16.80% ± 0.77% of cross-split within-parent pairs overlap and
52.62% ± 1.97% of test crops overlap pixels with a training crop. Both measurements
are 0.00% ± 0.00% under P2 and P3.

## Cx22 source overlap

The official Multi-Train and Multi-Test arms share no CCEDD parent. The separately
released Pair subset does share sources with both arms:

| Subset pair | Shared parents | Jaccard |
|---|---:|---:|
| Multi-Test / Multi-Train | 0 | 0.000 |
| Multi-Test / Pair | 55 | 0.126 |
| Multi-Train / Pair | 233 | 0.475 |

The released Cx22 archives contain generator metadata and label masks, rather than an
independent RGB `ImageDataSet.mat`. Consequently, the “two branches where both exist”
criterion is evaluated on SIPaKMeD. Cx22 instead receives source-existence, exact
generator-metadata, ROI-clipping, bbox-containment, and subset-overlap checks.

## Near-duplicate sensitivity grouping

The pHash+dHash consensus audit identifies 201 cross-parent candidate crop pairs.
Those candidates connect 170 parent IDs into 74 components (largest component: eight
parents), leaving 870 P3 groups. No pair is pixel-exact. These are explicitly labelled
thresholded near-duplicate candidates; they are used as a conservative P3 sensitivity
grouping and are not described as proven duplicate records.

At the frozen six-bit thresholds, a synthetic calibration detected all three injected
positive transformations and rejected all 100 deterministically selected crops from
other parents (precision 1.00, recall 1.00). This validates the implemented pHash+dHash
check on that calibration set only. CNN and domain-encoder duplicate detectors from the
broader project plan are not implemented here, so this release does not claim a
three-detector majority-vote audit.

## Evidence files

- `provenance/sipakmed_branch_a.parquet`
- `provenance/sipakmed_branch_b.parquet`
- `reports/generated/sipakmed_validation.csv`
- `reports/generated/sipakmed_bbox_iou.parquet`
- `provenance/cx22_parent_map.parquet`
- `reports/generated/cx22_validation.csv`
- `reports/generated/cx22_subset_overlap.csv`
- `provenance/cric_parent_map.parquet`
- `provenance/herlev_parent_map.parquet`
- `results/pixel_overlap_seed_level.csv`
- `results/duplicates/sipakmed_near_duplicate_pairs.csv`
- `results/duplicates/synthetic_detector_check.json`
