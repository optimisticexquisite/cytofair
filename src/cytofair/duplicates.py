"""CPU near-duplicate audit and P3 connected-component grouping."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def _bits_to_uint64(bits: np.ndarray) -> np.uint64:
    packed = np.packbits(bits.astype(np.uint8).ravel())
    return np.frombuffer(packed.tobytes(), dtype=">u8")[0].astype(np.uint64)


def image_hashes(path: str | Path) -> tuple[np.uint64, np.uint64, str]:
    values = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if values is None:
        raise ValueError(f"Cannot read image for hashing: {path}")
    resized = cv2.resize(values, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized)[:8, :8]
    phash = _bits_to_uint64(dct > np.median(dct[1:, :]))
    difference = cv2.resize(values, (9, 8), interpolation=cv2.INTER_AREA)
    dhash = _bits_to_uint64(difference[:, 1:] > difference[:, :-1])
    pixel_sha = hashlib.sha256(values.tobytes()).hexdigest()
    return phash, dhash, pixel_sha


def hash_manifest(provenance: pd.DataFrame) -> pd.DataFrame:
    required = {"crop_id", "parent_id", "crop_path"}
    missing = sorted(required - set(provenance.columns))
    if missing:
        raise ValueError(f"Duplicate audit misses: {missing}")
    records = []
    for row in provenance.itertuples(index=False):
        phash, dhash, pixel_sha = image_hashes(row.crop_path)
        records.append(
            {
                "crop_id": row.crop_id,
                "parent_id": row.parent_id,
                "phash": int(phash),
                "dhash": int(dhash),
                "pixel_sha256": pixel_sha,
            }
        )
    return pd.DataFrame(records)


def _hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def duplicate_pairs(
    hashes: pd.DataFrame, *, phash_threshold: int = 6, dhash_threshold: int = 6
) -> pd.DataFrame:
    required = {"crop_id", "parent_id", "phash", "dhash", "pixel_sha256"}
    missing = sorted(required - set(hashes.columns))
    if missing:
        raise ValueError(f"Hash manifest misses: {missing}")
    rows = list(hashes.itertuples(index=False))
    records = []
    for left_index, left in enumerate(rows):
        for right in rows[left_index + 1 :]:
            phash_distance = _hamming(int(left.phash), int(right.phash))
            dhash_distance = _hamming(int(left.dhash), int(right.dhash))
            exact = left.pixel_sha256 == right.pixel_sha256
            consensus = bool(exact or (phash_distance <= phash_threshold and dhash_distance <= dhash_threshold))
            if consensus:
                records.append(
                    {
                        "crop_id_a": left.crop_id,
                        "crop_id_b": right.crop_id,
                        "parent_id_a": left.parent_id,
                        "parent_id_b": right.parent_id,
                        "phash_distance": phash_distance,
                        "dhash_distance": dhash_distance,
                        "exact_pixels": exact,
                        "consensus_duplicate": True,
                        "cross_parent": left.parent_id != right.parent_id,
                    }
                )
    return pd.DataFrame(records)


class _UnionFind:
    def __init__(self, values):
        self.parent = {value: value for value in values}

    def find(self, value):
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            value, self.parent[value] = self.parent[value], root
        return root

    def union(self, left, right):
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            keep, merge = sorted((left_root, right_root))
            self.parent[merge] = keep


def attach_p3_groups(provenance: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    union = _UnionFind(provenance.parent_id.unique())
    if not pairs.empty:
        for pair in pairs.loc[pairs.consensus_duplicate & pairs.cross_parent].itertuples(index=False):
            union.union(pair.parent_id_a, pair.parent_id_b)
    out = provenance.copy()
    out["p3_group_id"] = out.parent_id.map(union.find)
    return out


def synthetic_detector_check(
    base_path: str | Path,
    output_dir: str | Path,
    *,
    negative_paths: Sequence[str | Path] = (),
    phash_threshold: int = 6,
    dhash_threshold: int = 6,
) -> dict[str, float | int]:
    """Validate transformed positives and unrelated negatives at fixed thresholds."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(base_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read synthetic source: {base_path}")
    transformed = {
        "identity": image,
        "brighter": cv2.convertScaleAbs(image, alpha=1.0, beta=20),
        "rescaled": cv2.resize(cv2.resize(image, (192, 192)), image.shape[1::-1]),
        "jpeg": image,
    }
    paths = []
    for name, values in transformed.items():
        path = output_dir / f"{name}.jpg"
        cv2.imwrite(str(path), values, [cv2.IMWRITE_JPEG_QUALITY, 75 if name == "jpeg" else 95])
        paths.append(path)
    base_hash = image_hashes(paths[0])

    def detected(path: str | Path) -> bool:
        candidate = image_hashes(path)
        return bool(
            _hamming(int(base_hash[0]), int(candidate[0])) <= phash_threshold
            and _hamming(int(base_hash[1]), int(candidate[1])) <= dhash_threshold
        )

    true_positive = 0
    for path in paths[1:]:
        true_positive += int(detected(path))
    false_positive = sum(int(detected(path)) for path in negative_paths)
    false_negative = len(paths) - 1 - true_positive
    true_negative = len(negative_paths) - false_positive
    precision_denominator = true_positive + false_positive
    return {
        "positive_examples": len(paths) - 1,
        "negative_examples": len(negative_paths),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": true_positive / precision_denominator if precision_denominator else 0.0,
        "recall": true_positive / (len(paths) - 1),
        "phash_threshold": phash_threshold,
        "dhash_threshold": dhash_threshold,
    }
