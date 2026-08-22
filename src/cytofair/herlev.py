"""Herlev negative-control manifest construction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

IMAGE_SUFFIXES = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def recover_one_cell_per_image(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    images = sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f"No Herlev images found under {root}")
    # The checked Zenodo package contains the 917 original BMPs plus 917 derived
    # `*INPUT.png` renderings. The released unit is the original single-cell BMP.
    originals = [path for path in images if path.suffix.lower() == ".bmp"]
    if originals:
        images = originals
    records = []
    for path in images:
        relative = path.relative_to(root).as_posix()
        records.append(
            {
                "crop_id": relative,
                "parent_id": relative,
                "class_name": path.parent.name,
                "crop_path": str(path.resolve()),
                "method": "one_released_image_per_parent",
                "match_confidence": 1.0,
            }
        )
    out = pd.DataFrame(records)
    if not out.crop_id.is_unique or not out.parent_id.is_unique:
        raise ValueError("Herlev control must have exactly one unique parent per image")
    return out
