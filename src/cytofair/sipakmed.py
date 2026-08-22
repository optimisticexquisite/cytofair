"""SIPaKMeD metadata parsing and independent image-based provenance recovery."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
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


def _best_match(
    crop: np.ndarray, parent: np.ndarray, *, scales: tuple[float, ...] = (1.0, 0.75, 1.25)
) -> tuple[float, int, int, int, int]:
    """Full-resolution NCC with multiscale fallback for release variants."""
    crop_h, crop_w = crop.shape[:2]
    parent_h, parent_w = parent.shape[:2]
    best = (-np.inf, 0, 0, crop_w, crop_h)
    for scale in scales:
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


def recover_branch_b(
    branch_a: pd.DataFrame,
    *,
    workers: int = 1,
    minimum_confidence: float = 0.95,
    minimum_margin: float = 0.02,
    coarse_scale: float = 0.25,
    top_k: int = 5,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int = 100,
) -> pd.DataFrame:
    """Independently identify and locate each crop among every same-class parent.

    Branch-A parent assignments are deliberately excluded from crop queries. Branch A
    supplies only the complete candidate-parent inventory and crop/class paths. A
    downscaled NCC pass ranks all parents; full-resolution NCC refines the top-k.
    """
    required = {"crop_id", "parent_id", "class_name", "crop_path", "parent_path"}
    missing = sorted(required - set(branch_a.columns))
    if missing:
        raise ValueError(f"Branch A table misses: {missing}")
    if not 0 < coarse_scale <= 1:
        raise ValueError("coarse_scale must be in (0, 1]")
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be positive")
    parents = branch_a[["parent_id", "class_name", "parent_path"]].drop_duplicates()
    if parents.parent_id.duplicated().any():
        raise ValueError("A parent_id maps to multiple candidate images")
    candidates = {
        class_name: group[["parent_id", "parent_path"]].to_dict("records")
        for class_name, group in parents.groupby("class_name", observed=True)
    }
    coarse_parents = {}
    for candidate in parents.itertuples(index=False):
        image = cv2.cvtColor(_read_rgb(candidate.parent_path), cv2.COLOR_BGR2GRAY)
        coarse_parents[candidate.parent_id] = cv2.resize(
            image, None, fx=coarse_scale, fy=coarse_scale, interpolation=cv2.INTER_AREA
        )

    # Same-class crops are adjacent in the deterministic release order. Keeping a
    # substantial but bounded parent working set avoids repeatedly decoding 9.4 MB
    # BMPs while remaining far below the host's available memory.
    @lru_cache(maxsize=128)
    def full_parent(path: str) -> np.ndarray:
        return _read_rgb(path)

    def match_one(row: dict) -> dict:
        crop = _read_rgb(row["crop_path"])
        coarse_crop = cv2.resize(
            cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY),
            None,
            fx=coarse_scale,
            fy=coarse_scale,
            interpolation=cv2.INTER_AREA,
        )
        ranked = []
        for candidate in candidates[row["class_name"]]:
            parent = coarse_parents[candidate["parent_id"]]
            if coarse_crop.shape[0] > parent.shape[0] or coarse_crop.shape[1] > parent.shape[1]:
                score = -np.inf
            else:
                response = cv2.matchTemplate(parent, coarse_crop, cv2.TM_CCOEFF_NORMED)
                score = float(response.max())
            ranked.append((score, candidate))
        ranked.sort(key=lambda item: item[0], reverse=True)
        def refine(scales: tuple[float, ...]):
            values = []
            for coarse_score, candidate in ranked[:top_k]:
                parent = full_parent(candidate["parent_path"])
                score, x, y, width, height = _best_match(crop, parent, scales=scales)
                values.append(
                    (score, coarse_score, candidate, x, y, width, height, parent.shape)
                )
            return values

        refined = refine((1.0,))
        # The released crops are normally exact pixel windows. Multiscale matching
        # remains a fail-safe for any post-processed/resized release variant.
        if max(item[0] for item in refined) < minimum_confidence:
            refined = refine((1.0, 0.75, 1.25))
        refined.sort(key=lambda item: item[0], reverse=True)
        best = refined[0]
        second_score = refined[1][0] if len(refined) > 1 else -1.0
        score, coarse_score, candidate, x, y, width, height, parent_shape = best
        parent_height, parent_width = parent_shape[:2]
        crop_height, crop_width = crop.shape[:2]
        return {
            **row,
            "parent_id": candidate["parent_id"],
            "parent_path": candidate["parent_path"],
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "crop_width": crop_width,
            "crop_height": crop_height,
            "parent_width": parent_width,
            "parent_height": parent_height,
            "coarse_match_confidence": coarse_score,
            "match_confidence": score,
            "second_best_confidence": second_score,
            "match_margin": score - second_score,
            "method": "template_ncc_all_same_class_parents",
        }

    crop_records = branch_a[["crop_id", "class_name", "crop_path"]].to_dict("records")
    matched = []
    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
    if checkpoint_path is not None and checkpoint_path.exists():
        partial = pd.read_parquet(checkpoint_path)
        if not set(partial.crop_id).issubset(set(branch_a.crop_id)) or partial.crop_id.duplicated().any():
            raise ValueError(f"Invalid Branch-B checkpoint: {checkpoint_path}")
        matched = partial.to_dict("records")
        completed = set(partial.crop_id)
        crop_records = [record for record in crop_records if record["crop_id"] not in completed]
        print(f"resuming image matching at {len(matched)}/{len(branch_a)} crops", flush=True)

    def collect(iterator):
        for result in iterator:
            matched.append(result)
            if len(matched) % checkpoint_every == 0 or len(matched) == len(branch_a):
                print(f"matched {len(matched)}/{len(branch_a)} crops", flush=True)
                if checkpoint_path is not None:
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(matched).to_parquet(checkpoint_path, index=False)

    if workers == 1:
        collect(map(match_one, crop_records))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            collect(pool.map(match_one, crop_records))
    out = pd.DataFrame(matched).sort_values("crop_id").reset_index(drop=True)
    out["accepted"] = (out.match_confidence >= minimum_confidence) & (
        out.match_margin >= minimum_margin
    )
    if checkpoint_path is not None:
        out.to_parquet(checkpoint_path, index=False)
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


def cross_split_pixel_overlap(
    provenance: pd.DataFrame, split: pd.DataFrame
) -> dict[str, int | float]:
    """Measure literal bbox overlap specifically across train and test."""
    pairs = bbox_iou_pairs(provenance)
    partitions = split.set_index("crop_id").partition
    if set(provenance.crop_id) != set(split.crop_id):
        raise ValueError("Pixel-overlap split must cover exactly the provenance crops")
    if pairs.empty:
        return {
            "cross_split_pairs": 0,
            "overlapping_cross_split_pairs": 0,
            "overlapping_cross_split_pair_fraction": 0.0,
            "test_crops_with_train_pixel_overlap": 0,
            "test_crop_pixel_overlap_fraction": 0.0,
        }
    pairs["partition_a"] = pairs.crop_id_a.map(partitions)
    pairs["partition_b"] = pairs.crop_id_b.map(partitions)
    cross = pairs.loc[pairs.partition_a != pairs.partition_b]
    overlapping = cross.loc[cross.intersection_pixels > 0]
    test_with_overlap = set(
        overlapping.crop_id_a.loc[overlapping.partition_a == "test"]
    ) | set(overlapping.crop_id_b.loc[overlapping.partition_b == "test"])
    n_test = int((split.partition == "test").sum())
    return {
        "cross_split_pairs": len(cross),
        "overlapping_cross_split_pairs": len(overlapping),
        "overlapping_cross_split_pair_fraction": len(overlapping) / len(cross)
        if len(cross)
        else 0.0,
        "test_crops_with_train_pixel_overlap": len(test_with_overlap),
        "test_crop_pixel_overlap_fraction": len(test_with_overlap) / n_test if n_test else 0.0,
    }
