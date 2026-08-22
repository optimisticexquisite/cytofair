# Title-to-results claim map

Title: **Do We Train on Test Cells? Provenance Leakage in Public Cervical Cytology
Benchmarks**

| Title claim | Corresponding results row or evidence |
|---|---|
| Do we train on test cells? | Table 1 reports P1 test crops whose recovered parent occurs in training; the pixel-overlap result reports test crops sharing literal pixels with a training crop. The wording is a question, not a claim that identical crop files occur in both arms. |
| Provenance | The SIPaKMeD Branch-A/Branch-B row reports 4,049/4,049 agreement and bbox containment; Cx22 reports exact generator-source recovery. |
| Leakage | Table 1 reports analytic and observed P1 parent straddling, leaked-test-crop fractions, and zero P2/P3 straddling. |
| Public | Dataset rows refer only to the declared owner releases in `data_sources.json`; no private cohort result appears. |
| Cervical cytology | Table 1 contains SIPaKMeD, Cx22, CRIC, and the Herlev negative control, all named as cervical-cytology datasets in the data section. |
| Benchmarks | Table 1 contains one row per audited benchmark rather than extrapolating from a single release. |

This mapping supports `title_adjectives_have_rows=true` in the paper checklist. It
does not make the paper ready while the independent reader, lab-paper crosswalk, or
clean-clone gates remain open.

