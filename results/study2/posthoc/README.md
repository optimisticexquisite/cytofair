# Post hoc reanalyses of Study 1 (exploratory, not confirmatory)

These files apply the Study 2 H1 model (`analysis/study2/protocol_lmm.R`) to frozen Study 1
runs. The model was chosen after the Study 1 data were seen, so these results motivate
Study 2 and supply its pilot variance components; they are not evidence for any Study 2
hypothesis.

| File | Data | Contrast |
|---|---|---|
| `study1_sipakmed_pooled_lmm.json` | 6 classifiers x 5 seeds (Study 1 seeds 17-101) | SIPaKMeD accuracy, P1 - P2 |
| `study1_cx22_semantic_pooled_lmm.json` | 5 semantic segmentation models x 5 seeds | pooled Cx22 macro Dice, P1 - P2 |

The two Cx22 instance-segmentation models are too few for an architecture random slope
and are summarised descriptively in the project record instead.
