"""CRIC cell-coordinate provenance recovery from the authors' Figshare table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

EXPECTED_CROPS = 11_534
EXPECTED_PARENTS = 400


def recover_classifications(path: str | Path) -> pd.DataFrame:
    source = pd.read_csv(path)
    required = {
        "image_id",
        "image_filename",
        "image_doi",
        "cell_id",
        "bethesda_system",
        "nucleus_x",
        "nucleus_y",
    }
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"CRIC classifications miss columns: {missing}")
    if len(source) != EXPECTED_CROPS or source.image_id.nunique() != EXPECTED_PARENTS:
        raise ValueError(
            f"Unexpected CRIC release size: crops={len(source)}, "
            f"parents={source.image_id.nunique()}"
        )
    if source.duplicated(["image_id", "cell_id"]).any():
        raise ValueError("Duplicate CRIC image/cell key")
    if source.groupby("image_id").image_filename.nunique().max() != 1:
        raise ValueError("A CRIC image_id maps to multiple filenames")
    out = pd.DataFrame(
        {
            "crop_id": source.apply(
                lambda row: f"{int(row.image_id):04d}/{int(row.cell_id):05d}", axis=1
            ),
            "parent_id": source.image_id.map(lambda value: f"cric-{int(value):04d}"),
            "class_name": source.bethesda_system.astype("string"),
            "source_name": source.image_filename.astype("string"),
            "source_doi": source.image_doi.astype("string"),
            "center_x": source.nucleus_x.astype(int),
            "center_y": source.nucleus_y.astype(int),
            "method": "published_cell_coordinates",
            "match_confidence": 1.0,
        }
    )
    if not out.crop_id.is_unique:
        raise ValueError("CRIC canonical crop IDs are not unique")
    return out.sort_values("crop_id").reset_index(drop=True)

