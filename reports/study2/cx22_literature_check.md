# Do published Cx22 studies pool the release subsets?

**Question (roadmap, Part 5).** Cx22's official Multi-Train / Multi-Test partition is
parent-disjoint, but the separately released Pair subset shares 233 CCEDD source images
with Multi-Train and 55 with Multi-Test. Pooling the 1,320 released ROIs and re-splitting
them at the ROI level therefore reintroduces source-image leakage. Is that risk realised in
published work, or only theoretical?

**Method (2026-10-18).** Europe PMC search `"Cx22" AND (cervical OR cytology)` (43 hits),
then the open-access full text (Europe PMC `fullTextXML`) of every hit that uses Cx22
images, searched for sentences describing training, testing, splitting, or folds. Hits
where "Cx22" denotes something else (a connexin, a compound) were discarded. Papers without
open full text were not assessed.

**Stated protocols (verbatim excerpts).** The table records what each paper states; it
does not infer anything the text does not say.

| Paper | Use of Cx22 | Stated protocol |
|---|---|---|
| BMC Medical Imaging 2023, deep ensemble segmentation (PMC10514950) | train and test | "The Cx22 dataset contained a training dataset and a testing dataset with 400 and 100 samples, respectively." Official Multi split. |
| Scientific Reports 2025, Mask2Former with denoising (PMC12095581) | train and test | "The Cx22 dataset contains 400 images in the training dataset and 100 ones in the testing dataset." Official Multi split. |
| Cell Reports Medicine 2026, UniCAS foundation model (PMC12866161) | segmentation benchmark | "Cx22-multi: This public dataset consists of 500 cytology images, divided into 400 for training and 100 for testing." Official Multi split. |
| Cancer Cell International 2026, joint U-Net and CNN ensemble (PMC13041069) | train and validate | Describes Cx22 as "1,320 images" and states "a stratified five-fold cross-validation protocol, in which the data were divided into five mutually exclusive folds", with splitting "at the image level, ensuring that all cell instances derived from a single image remained within the same subset". It does not state that Pair was excluded or that CCEDD source images were kept within one fold. The same paper treats SIPaKMeD crops as "independent, pre-cropped single-cell samples" and separates them "at the image level". |
| Scientific Data 2024 (PMC11227563); Cureus 2025 (PMC11885184) | cited only | No Cx22 training or evaluation. |
| Frontiers in Oncology 2025 (PMC12929100) | unclear | Describes "Cx22 clinical data" modelled with an MLP; the text does not match the image dataset and was not assessed further. |

**Answer.** Of four open-access studies that train or evaluate on Cx22 images, three use
the official 400/100 Multi-Train/Multi-Test split, which is parent-disjoint. One describes
image-level five-fold cross-validation over the 1,320-image collection without stating that
the Pair subset or the CCEDD source images were kept apart, which is the downstream reuse
pattern the Cx22 stress test models. The risk is therefore realised in at least one recent
publication, while most users follow the safe official split, consistent with Cx22 serving
as an existence proof that a hierarchy-aware release is used in practice. The same study's
description of SIPaKMeD crops as independent samples is the misconception this project
addresses.

**Limits.** Open-access full texts only; closed-access Cx22 users (for example a 2026
Scientific Reports background-gating network, a 2024 Computers in Biology and Medicine
unsupervised instance-segmentation method, and a 2025 Medical & Biological Engineering &
Computing method) were not assessed. Whether a study pooled subsets can be stated only from
what the paper reports.

**Reporting choice.** In keeping with the project's rule that no individual paper is named
as deficient, the manuscript reports these counts in aggregate. This file keeps the
verbatim evidence so the count can be checked.
