# Generated paper tables

All stochastic cells are mean ± sample SD over at least five fixed seeds.

## Table 1. Structural leakage

| Dataset | Crops | Parents | Analytic P1 straddle | Observed P1 straddle | P1 test crops with train parent | P2 straddle | P3 straddle |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SIPaKMeD | 4,049 | 966 | 44.83% | 45.63 ± 1.04% | 92.62 ± 0.40% | 0.00 ± 0.00% | 0.00 ± 0.00% |
| Cx22 | 1,320 | 514 | 33.93% | 34.01 ± 1.10% | 85.23 ± 3.51% | 0.00 ± 0.00% | 0.00 ± 0.00% |
| CRIC | 11,534 | 400 | 94.88% | 94.35 ± 1.10% | 99.97 ± 0.08% | 0.00 ± 0.00% | 0.00 ± 0.00% |
| Herlev | 917 | 917 | 0.00% | 0.00 ± 0.00% | 0.00 ± 0.00% | 0.00 ± 0.00% | 0.00 ± 0.00% |

## Table 2a. Matched classification accuracy

| Dataset | Architecture | P1 | P1′ | P2 | P3 |
| --- | --- | --- | --- | --- | --- |
| SIPaKMeD | densenet121 | 0.9812 ± 0.0028 | 0.9812 ± 0.0028 | 0.9672 ± 0.0069 | 0.9738 ± 0.0050 |
| SIPaKMeD | efficientnet_b0 | 0.9778 ± 0.0049 | 0.9778 ± 0.0049 | 0.9669 ± 0.0149 | 0.9741 ± 0.0071 |
| SIPaKMeD | mobilenet_v2 | 0.9741 ± 0.0045 | 0.9741 ± 0.0045 | 0.9516 ± 0.0150 | 0.9620 ± 0.0093 |
| SIPaKMeD | resnet18 | 0.9719 ± 0.0059 | 0.9719 ± 0.0059 | 0.9600 ± 0.0089 | 0.9635 ± 0.0128 |
| SIPaKMeD | resnet50 | 0.9751 ± 0.0047 | 0.9751 ± 0.0047 | 0.9679 ± 0.0042 | 0.9627 ± 0.0053 |
| SIPaKMeD | vit_b_16 | 0.9714 ± 0.0095 | 0.9714 ± 0.0095 | 0.9556 ± 0.0084 | 0.9553 ± 0.0168 |

## Table 2b. Preregistered paired inference

| Dataset | Architecture | Contrast | Paired Δ, mean ± SD | Test | Inference |
| --- | --- | --- | --- | --- | --- |
| SIPaKMeD | densenet121 | P1−P2 | 0.0141 ± 0.0090 | paired_t | p=0.0251, q=0.1109; cohens_dz=1.561 [0.985, 4.747] |
| SIPaKMeD | efficientnet_b0 | P1−P2 | 0.0109 ± 0.0185 | paired_t | p=0.2591, q=0.2591; cohens_dz=0.588 [-0.316, 4.668] |
| SIPaKMeD | mobilenet_v2 | P1−P2 | 0.0225 ± 0.0163 | paired_t | p=0.0370, q=0.1109; cohens_dz=1.377 [0.809, 6.686] |
| SIPaKMeD | resnet18 | P1−P2 | 0.0119 ± 0.0128 | paired_t | p=0.1081, q=0.1511; cohens_dz=0.922 [0.442, 2.365] |
| SIPaKMeD | resnet50 | P1−P2 | 0.0072 ± 0.0083 | paired_t | p=0.1260, q=0.1511; cohens_dz=0.863 [0.085, 4.025] |
| SIPaKMeD | vit_b_16 | P1−P2 | 0.0158 ± 0.0170 | paired_t | p=0.1059, q=0.1511; cohens_dz=0.931 [0.057, 4.950] |
| SIPaKMeD | densenet121 | P1′−P2 | 0.0141 ± 0.0090 | paired_t | p=0.0251, q=0.1109; cohens_dz=1.561 [0.932, 4.747] |
| SIPaKMeD | efficientnet_b0 | P1′−P2 | 0.0109 ± 0.0185 | paired_t | p=0.2591, q=0.2591; cohens_dz=0.588 [-0.316, 4.610] |
| SIPaKMeD | mobilenet_v2 | P1′−P2 | 0.0225 ± 0.0163 | paired_t | p=0.0370, q=0.1109; cohens_dz=1.377 [0.809, 6.686] |
| SIPaKMeD | resnet18 | P1′−P2 | 0.0119 ± 0.0128 | paired_t | p=0.1081, q=0.1511; cohens_dz=0.922 [0.426, 2.442] |
| SIPaKMeD | resnet50 | P1′−P2 | 0.0072 ± 0.0083 | paired_t | p=0.1260, q=0.1511; cohens_dz=0.863 [0.085, 4.025] |
| SIPaKMeD | vit_b_16 | P1′−P2 | 0.0158 ± 0.0170 | paired_t | p=0.1059, q=0.1511; cohens_dz=0.931 [0.120, 4.950] |

## Table 2c. Secondary P2−P3 sensitivity contrast

These p-values are unadjusted because P2−P3 is outside the stated confirmatory BH family. Every p-value retains its paired effect and 95% CI.

| Dataset | Architecture | Contrast | Paired Δ, mean ± SD | Test | Inference |
| --- | --- | --- | --- | --- | --- |
| SIPaKMeD | densenet121 | P2−P3 | -0.0067 ± 0.0069 | paired_t | unadjusted p=0.0971; cohens_dz=-0.965 [-3.266, -0.725] |
| SIPaKMeD | efficientnet_b0 | P2−P3 | -0.0072 ± 0.0194 | paired_t | unadjusted p=0.4562; cohens_dz=-0.369 [-2.708, 0.467] |
| SIPaKMeD | mobilenet_v2 | P2−P3 | -0.0104 ± 0.0138 | paired_t | unadjusted p=0.1673; cohens_dz=-0.753 [-3.329, 0.075] |
| SIPaKMeD | resnet18 | P2−P3 | -0.0035 ± 0.0119 | paired_t | unadjusted p=0.5510; cohens_dz=-0.291 [-1.516, 1.255] |
| SIPaKMeD | resnet50 | P2−P3 | 0.0052 ± 0.0066 | paired_t | unadjusted p=0.1545; cohens_dz=0.784 [0.361, 2.534] |
| SIPaKMeD | vit_b_16 | P2−P3 | 0.0002 ± 0.0234 | paired_t | unadjusted p=0.9823; cohens_dz=0.011 [-3.338, 0.817] |

## Table 2d. Registered protocol-ordering check

| Dataset | Architecture | Registered P3 ≤ P2 ≤ P1 |
| --- | --- | --- |
| SIPaKMeD | densenet121 | Violation |
| SIPaKMeD | efficientnet_b0 | Violation |
| SIPaKMeD | mobilenet_v2 | Violation |
| SIPaKMeD | resnet18 | Violation |
| SIPaKMeD | resnet50 | Pass |
| SIPaKMeD | vit_b_16 | Pass |

## Table 2e. Architecture-ranking stability

The ranking p-value is secondary and outside the confirmatory BH family.

| Dataset | Rankings | Architectures | Kendall τ [seed-bootstrap 95% CI] | Inference | Reordered pairs | Tie changes |
| --- | --- | --- | --- | --- | --- | --- |
| SIPaKMeD | P1 vs P2 | 6 | 0.467 [0.071, 0.733] | unadjusted p=0.2722 | 4/15 | 0 |
