"""SIPaKMeD metadata parsing and independent image-based provenance recovery."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

IMAGE_SUFFIXES = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
CLASS_ALIASES = {
    "superficial-intermediate": "Superficial-Intermediate",
    "superficial_intermediate": "Superficial-Intermediate",
    "parabasal": "Parabasal",
    "koilocytotic": "Koilocytotic",
    "dyskeratotic": "Dyskeratotic",
    "dysketarotic": "Dyskeratotic",
    "metaplastic": "Metaplastic",
}


def _class_name(path: Path) -> str:
    lowered = "/".join(part.lower().removeprefix("im_") for part in path.parts)
    for token, canonical in CLASS_ALIASES.items():
        if token in lowered:
            return canonical
    raise ValueError(f"Cannot infer SIPaKMeD class from {path}")


def _image_paths(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def discover_release(root: str | Path) -> tuple[list[Path], list[Path]]:
    """Discover crops below CROPPED directories and cluster images elsewhere."""
    root = Path(root)
    images = _image_paths(root)
    crops = [path for path in images if any(part.lower() == "cropped" for part in path.parts)]
    parents = [path for path in images if not any(part.lower() == "cropped" for part in path.parts)]
    if not crops or not parents:
        raise ValueError(
            f"Expected images both inside and outside CROPPED directories; "
            f"found {len(crops)} crops and {len(parents)} parents under {root}"
        )
    return crops, parents


def parent_stem_from_crop(crop_stem: str) -> str:
    """Map the released `001_01` crop convention to cluster stem `001`."""
    match = re.fullmatch(r"(.+?)_(\d+)", crop_stem)
    if not match:
        raise ValueError(f"Crop filename does not encode a terminal cell index: {crop_stem}")
    return match.group(1)


def recover_branch_a(root: str | Path) -> pd.DataFrame:
    """Recover parents directly and exactly from release filenames."""
    root = Path(root)
    crops, parents = discover_release(root)
    parent_lookup: dict[tuple[str, str], Path] = {}
    for parent in parents:
        key = (_class_name(parent), parent.stem)
        if key in parent_lookup:
            raise ValueError(f"Duplicate parent stem within class: {key}")
        parent_lookup[key] = parent

    records: list[dict] = []
    for crop in crops:
        class_name = _class_name(crop)
        parent_stem = parent_stem_from_crop(crop.stem)
        key = (class_name, parent_stem)
        parent = parent_lookup.get(key)
        if parent is None:
            raise ValueError(f"No parent {key} for crop {crop}")
        records.append(
            {
                "crop_id": crop.relative_to(root).as_posix(),
                "parent_id": f"{class_name}/{parent_stem}",
                "class_name": class_name,
                "crop_path": str(crop.resolve()),
                "parent_path": str(parent.resolve()),
                "method": "filename",
                "match_confidence": 1.0,
            }
        )
    return pd.DataFrame(records).sort_values("crop_id").reset_index(drop=True)


def _read_rgb(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV cannot read {path}")
    return image


def _best_match(crop: np.ndarray, parent: np.ndarray) -> tuple[float, int, int, int, int]:
    """Full-resolution NCC with multiscale fallback for release variants."""
    crop_h, crop_w = crop.shape[:2]
    parent_h, parent_w = parent.shape[:2]
    best = (-np.inf, 0, 0, crop_w, crop_h)
    for scale in (1.0, 0.75, 1.25):
        scaled_w = max(8, round(crop_w * scale))
        scaled_h = max(8, round(crop_h * scale))
        if scaled_w > parent_w or scaled_h > parent_h:
            continue
        candidate = crop if scale == 1.0 else cv2.resize(crop, (scaled_w, scaled_h))
        response = cv2.matchTemplate(parent, candidate, cv2.TM_CCOEFF_NORMED)
        _, score, _, location = cv2.minMaxLoc(response)
        if score > best[0]:
            best = (float(score), int(location[0]), int(location[1]), scaled_w, scaled_h)
    return best


def _match_one(row: dict) -> dict:
    crop = _read_rgb(row["crop_path"])
    parent = _read_rgb(row["parent_path"])
    score, x, y, width, height = _best_match(crop, parent)
    parent_height, parent_width = parent.shape[:2]
    crop_height, crop_width = crop.shape[:2]
    return {
        **row,
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "crop_width": crop_width,
        "crop_height": crop_height,
        "parent_width": parent_width,
        "parent_height": parent_height,
        "match_confidence": score,
        "method": "template_ncc_within_filename_parent",
    }


def recover_branch_b(
    branch_a: pd.DataFrame,
    *,
    workers: int = 1,
    minimum_confidence: float = 0.95,
) -> pd.DataFrame:
    """Independently locate each crop inside its Branch-A candidate parent.

    The parent candidate is constrained by class and filename, but the acceptance
    evidence (pixel location and NCC) is image-derived. A later exhaustive matcher can
    challenge candidate identity on ambiguous/failed rows without wasting O(4049*200)
    comparisons for exact release files.
    """
    required = {"crop_id", "parent_id", "class_name", "crop_path", "parent_path"}
    missing = sorted(required - set(branch_a.columns))
    if missing:
        raise ValueError(f"Branch A table misses: {missing}")
    records = branch_a[list(required)].to_dict("records")
    if workers == 1:
        matched = [_match_one(record) for record in records]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            matched = list(pool.map(_match_one, records))
    out = pd.DataFrame(matched).sort_values("crop_id").reset_index(drop=True)
    out["accepted"] = out.match_confidence >= minimum_confidence
    return out


def bbox_iou_pairs(provenance: pd.DataFrame) -> pd.DataFrame:
    """Calculate all within-parent bbox IoUs once Branch B has run."""
    required = {"crop_id", "parent_id", "x", "y", "width", "height"}
    missing = sorted(required - set(provenance.columns))
    if missing:
        raise ValueError(f"BBox table misses: {missing}")
    records: list[dict] = []
    for parent_id, group in provenance.groupby("parent_id", observed=True):
        rows = list(group.itertuples(index=False))
        for left_index, left in enumerate(rows):
            for right in rows[left_index + 1 :]:
                lx2, ly2 = left.x + left.width, left.y + left.height
                rx2, ry2 = right.x + right.width, right.y + right.height
                intersection = max(0, min(lx2, rx2) - max(left.x, right.x)) * max(
                    0, min(ly2, ry2) - max(left.y, right.y)
                )
                union = left.width * left.height + right.width * right.height - intersection
                records.append(
                    {
                        "parent_id": parent_id,
                        "crop_id_a": left.crop_id,
                        "crop_id_b": right.crop_id,
                        "intersection_pixels": int(intersection),
                        "iou": float(intersection / union) if union else 0.0,
                    }
                )
    return pd.DataFrame(records)

