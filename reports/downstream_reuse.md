# What CytoFair makes reusable

This note maps the implemented assets to the follow-on work in the project plan. It
describes technical reuse; it does not claim that an unrun study or an unwritten paper
already exists.

## Deliberately partitioned publication units

- **ISBI unit:** exact provenance recovery, structural leakage, the parent-identity
  mechanism probe, matched P1/P1′/P2/P3 classification, fixed split files, and the
  raw-to-Table-1 command.
- **Journal extension:** the separately scoped annotation study, broader
  cross-dataset validation, expanded detector ensemble, and learning-curve analysis.
  These are not silently folded into the conference result and remain future work
  until their own protocols and evidence exist.

## Projects unblocked by the apparatus

| Project | Reused asset | What no longer has to be rebuilt |
|---|---|---|
| F3: Cx22 seed robustness | `cx22_parent_map.parquet`, fixed-seed split generator, run JSON schema, mean/SD and paired-inference code | source-image grouping, seed ledger, and re-scoring scaffold |
| F8: cross-dataset transfer matrix | canonical parent maps, `classification_split`, model/runtime metadata, binary-label crosswalk slot | leakage-safe source folds and experiment provenance |
| F2: foundation encoders on small sets | fixed P1/P1′/P2/P3 CSVs and the architecture-neutral result summarizer | fair split comparison, seed accounting, BH/effect-size reporting |

The split CSVs are the stable interface: downstream code joins on `crop_id` and never
reconstructs a split from filenames or row order. Each exported split has an adjacent
SHA-256 manifest so a later paper can identify the exact evaluation asset.

## Second-domain V2

The same audit has a direct specimen grouping interpretation in Fish-Pak (three views
per specimen) and a temporal grouping interpretation in Croatian Fish (frames sampled
from continuous video). The reusable sequence is:

1. recover specimen or video-episode parent IDs;
2. measure crop/frame-level straddling and compare it with the analytic prediction;
3. create parent- or episode-disjoint fixed splits;
4. run the identity probe and the P1/P1′/P2/P3 harness;
5. release split hashes and raw seed-level results.

No fish-domain result is claimed here. This repository supplies the audit contract and
the machinery needed to make that follow-on a short adaptation rather than a new
experimental system.

## Citation-bearing asset

The durable output is the split/provenance interface, not a renamed dataset. A future
paper can cite CytoFair while continuing to cite the original dataset owners and can
reproduce the exact parent-disjoint assignment from a versioned file hash. Adoption is
an intended outcome, not a citation forecast presented as an observed result.
