#!/usr/bin/env python3
"""Check rendered transplant plans against the committed plans; export background fractions.

The renderer recomputes each plan from the committed 2C design. This script
fails unless every rendered plan matches the committed plan row for row, then
writes the per-recipient background fraction (a deterministic image property,
computed before any model exists) next to the committed plans.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rendered", help="directory holding transplant_seed*_plan.csv from the renderer")
    parser.add_argument("--committed", default="splits/study2/sipakmed_2b")
    args = parser.parse_args()
    committed_root = Path(args.committed)
    for rendered_path in sorted(Path(args.rendered).glob("transplant_seed*_plan.csv")):
        seed = int(rendered_path.stem.removeprefix("transplant_seed").removesuffix("_plan"))
        rendered = pd.read_csv(rendered_path)
        committed = pd.read_csv(committed_root / f"sipakmed_2b_plan_seed{seed}.csv")
        columns = sorted(committed.columns)
        left = committed[columns].sort_values("composite_id").reset_index(drop=True)
        right = rendered[columns].sort_values("composite_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(left, right, check_dtype=False)
        fractions = rendered.drop_duplicates("recipient_id")[["recipient_id", "background_fraction"]]
        fractions.sort_values("recipient_id").to_csv(
            committed_root / f"background_fraction_seed{seed}.csv", index=False, lineterminator="\n"
        )
        print(f"seed {seed}: plan verified ({len(rendered)} composites), {len(fractions)} recipients")


if __name__ == "__main__":
    main()
