#!/usr/bin/env python3
"""Numbers used only by the ISBI paper, derived from the committed follow-up tables.

Writes ``generated/derived.tex``:
* ``\\LeaceAboveFloorCut``: share of the above-floor within-class parent recoverability removed
  by rank-256 LEACE, (R_0 - R_256) / (R_0 - floor), with R averaged over all twelve
  architectures, both protocols and all seeds (the same averages as LeaceRzero/LeaceRtop).

Run from the repository root:  python3 paper/isbi2027/derived_numbers.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "generated" / "derived.tex"


def main() -> None:
    leace = pd.read_csv(ROOT / "results/study2/followup/tables/leace_runs.csv")
    recoverability = 100 * leace.groupby("rank").within_class_accuracy.mean()
    floor = 100 * leace.within_class_floor.iloc[0]
    top = recoverability.index.max()
    cut = 100 * (recoverability[0] - recoverability[top]) / (recoverability[0] - floor)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(f"\\newcommand{{\\LeaceAboveFloorCut}}{{{cut:.0f}}}\n")
    print(OUT, f"LeaceAboveFloorCut={cut:.1f}")


if __name__ == "__main__":
    main()
