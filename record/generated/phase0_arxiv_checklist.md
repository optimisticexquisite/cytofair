# Phase 0: arXiv v1 readiness

Status of each pre-posting item from the scaling roadmap, as checked on 2026-10-18.
"Verified" means checked directly; "Authors" means only the authors can do it.

| Item | Status | Evidence or action |
|---|---|---|
| Public repository installs and tests pass | Verified | Fresh `git clone` of `github.com/optimisticexquisite/cytofair` at `2c5d5c3`: `uv sync --extra dev` succeeds, 45/45 tests pass, `cytofair --help` runs. |
| Repository ships provenance maps and split manifests | Verified | `provenance/*.parquet` (SIPaKMeD Branch A/B, P3 map, Cx22, CRIC, Herlev) and `splits/generated/` (20 SIPaKMeD split CSVs + manifest) are tracked. |
| Repository contains the work the paper reports | **Gap** | The public `main` stops at the 22 August results. The 11 Cx22-extension commits and all Study 2 work exist only locally (branch `study2`). Push before posting so the paper's repository link resolves to the reported code. |
| Software licence | **Authors** | The repository has no `LICENSE` file, so the toolkit is legally all-rights-reserved and cannot be reused. Choose a code licence (MIT or Apache-2.0 are common) and state that derived artifacts follow each source's terms (below). |
| Redistribution terms for derived artifacts | **Authors** (partly verified) | No source images are redistributed. Derived parent IDs, coordinates, hashes, and split assignments: CRIC and the Herlev repackaging are CC BY 4.0 (attribution suffices); SIPaKMeD is "public for experimental use, cite Plissiti et al." (derived IDs very likely fine; confirm); Cx22 is non-commercial research and education only; CCEDD forbids redistribution of the images (the released maps contain only source file names and coordinates; confirm this is acceptable to the CCEDD authors). |
| CRIC images for Study 2 | Verified | All 400 slide images are CC BY 4.0 on Figshare (collection 4960286) and were fetched with Figshare MD5 checks; `data_manifests/cric_images_verified.json` records MD5 and SHA-256 per file. Images are not committed. |
| All co-authors approve the exact posted version | **Authors** | Required before posting. The current draft still has e-mail and repository-URL placeholders. |
| PDF text extraction | Fixed in sources | The project record's extracted text shows broken small caps ("CytoF air"), ligature glyphs, and split equations. The Study 2 manuscript preamble sets T1 font encoding and Unicode glyph mapping; the same two lines are suggested for the ISBI draft (see `paper/v2/README.md`). |
| "Replication in preparation" sentence | Suggested text | "A preregistered, adequately powered replication and a causal decomposition of the effect are in progress." (Study 2 is registered in this repository as `PREREGISTRATION_STUDY2.md`.) |
| Anonymity policy of the target venue | **Authors** | Posting a preprint is usually allowed; promoting it during double-blind review may not be. Check the chosen venue's policy before announcing the preprint. |

## Note on public timestamps

The Study 2 preregistration is committed locally before any Study 2 result exists, matching
how Study 1 was registered (its plan commit predated its first result commit, and both were
pushed later). A local commit is weak evidence of timing. For an independent timestamp,
push the registration commit, or deposit `PREREGISTRATION_STUDY2.md` on OSF, as early as
possible; the commit hash and file hashes in `reports/study2/registration_hashes.txt` let a
reader confirm that the deposited text is the one that governed the analysis.
