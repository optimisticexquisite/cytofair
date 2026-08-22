"""Recover Cx22 crop provenance from the authors' MATLAB generator metadata."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def _string_object(path: Path, key: str) -> list[str]:
    """Decode MATLAB's MCOS-backed string array used by the Cx22 release.

    MATLAB stores the public variable as an opaque six-word handle. Its payload in
    `/#refs#/c` is: version, ndim, dimensions, one uint64 length per string, then four
    packed UTF-16LE code units per uint64 word. The consistency checks below make this
    reverse-engineered reader fail closed if a future release uses a different layout.
    """
    with h5py.File(path, "r") as handle:
        dataset = handle[key]
        if dataset.attrs.get("MATLAB_class") != np.bytes_(b"string"):
            raise ValueError(f"{path}:{key} is not a MATLAB string object")
        payload = np.asarray(handle["#refs#/c"]).ravel()
    version = int(payload[0])
    ndim = int(payload[1])
    if version != 1 or ndim <= 0 or ndim > 8:
        raise ValueError(f"Unsupported MATLAB string payload header: {payload[:4]}")
    dimensions = payload[2 : 2 + ndim].astype(int)
    count = int(np.prod(dimensions))
    lengths = payload[2 + ndim : 2 + ndim + count].astype(int)
    packed = payload[2 + ndim + count :].astype("<u8", copy=False)
    expected_words = (int(lengths.sum()) + 3) // 4
    if len(packed) != expected_words:
        raise ValueError(f"String payload length mismatch: {len(packed)} != {expected_words}")
    decoded = packed.tobytes().decode("utf-16le")[: int(lengths.sum())]
    values, position = [], 0
    for length in lengths:
        values.append(decoded[position : position + length])
        position += int(length)
    return values


def _numeric(path: Path, key: str) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        values = np.asarray(handle[key])
    # MATLAB v7.3 dimensions are exposed in HDF5's reverse order.
    return values.T


def _generator_dir(root: Path) -> Path:
    candidates = sorted(path.parent for path in root.rglob("ImageDataNames.mat"))
    if len(candidates) != 1:
        raise ValueError(f"Expected one Cx22 generator directory under {root}; found {candidates}")
    return candidates[0]


def recover_metadata(root: str | Path, subset: str | None = None) -> pd.DataFrame:
    """Create one row per generated Cx22 ROI with exact source name and coordinates."""
    root = Path(root)
    generator = _generator_dir(root)
    subset = subset or generator.parent.name
    names = _string_object(generator / "ImageDataNames.mat", "ImageDataNames")
    sizes = _numeric(generator / "ROIs_W_H.mat", "ROIs_W_H")
    origins = _numeric(generator / "ROIs_x_y.mat", "ROIs_x_y")
    if sizes.ndim != 2 or sizes.shape[1] != 2:
        raise ValueError(f"Unexpected ROIs_W_H shape: {sizes.shape}")
    if origins.shape != sizes.shape or len(names) != len(sizes):
        raise ValueError(
            f"Cx22 metadata length mismatch: names={len(names)}, sizes={sizes.shape}, "
            f"origins={origins.shape}"
        )
    records = []
    for index, (name, size, origin) in enumerate(zip(names, sizes, origins, strict=True)):
        width, height = (round(value) for value in size)
        x, y = (round(value) for value in origin)
        records.append(
            {
                "crop_id": f"{subset}/{index:04d}",
                "parent_id": Path(name).stem,
                "class_name": "segmentation",
                "subset": subset,
                "source_name": f"{Path(name).stem}.png",
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "method": "generator_metadata",
                "match_confidence": 1.0,
            }
        )
    return pd.DataFrame(records)


def recover_all(root: str | Path, image_root: str | Path | None = None) -> pd.DataFrame:
    root = Path(root)
    generator_files = sorted(root.rglob("ImageDataNames.mat"))
    if not generator_files:
        raise ValueError(f"No Cx22 generator metadata found under {root}")
    frames = [recover_metadata(path.parent.parent, path.parent.parent.name) for path in generator_files]
    out = pd.concat(frames, ignore_index=True)
    if not out.crop_id.is_unique:
        raise ValueError("Cx22 crop IDs are not unique across subsets")
    out = out.sort_values("crop_id").reset_index(drop=True)
    if image_root is not None:
        out = add_parent_dimensions(out, image_root)
    return out


def subset_parent_overlap(frame: pd.DataFrame) -> pd.DataFrame:
    """Pairwise source-parent overlap between Cx22 release subsets."""
    parents = {
        subset: set(group.parent_id.astype(str))
        for subset, group in frame.groupby("subset", observed=True)
    }
    records = []
    names = sorted(parents)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            common = parents[left] & parents[right]
            union = parents[left] | parents[right]
            records.append(
                {
                    "subset_a": left,
                    "subset_b": right,
                    "parents_a": len(parents[left]),
                    "parents_b": len(parents[right]),
                    "shared_parents": len(common),
                    "jaccard": len(common) / len(union) if union else 0.0,
                }
            )
    return pd.DataFrame(records)


def add_parent_dimensions(frame: pd.DataFrame, image_root: str | Path) -> pd.DataFrame:
    """Attach real CCEDD dimensions without loading entire images into memory."""
    from PIL import Image

    image_root = Path(image_root)
    paths = sorted(image_root.rglob("*.png"))
    lookup = {path.name: path for path in paths}
    if len(lookup) != len(paths):
        raise ValueError("CCEDD source image names are not unique")
    out = frame.copy()
    widths, heights, parent_paths = [], [], []
    for name in out.source_name:
        path = lookup.get(name)
        if path is None:
            widths.append(np.nan)
            heights.append(np.nan)
            parent_paths.append(None)
            continue
        with Image.open(path) as image:
            width, height = image.size
        widths.append(width)
        heights.append(height)
        parent_paths.append(str(path.resolve()))
    out["parent_width"] = widths
    out["parent_height"] = heights
    out["parent_path"] = parent_paths
    if out[["parent_width", "parent_height"]].isna().any().any():
        missing = sorted(out.loc[out.parent_width.isna(), "source_name"].unique())
        raise ValueError(f"Missing CCEDD source images: {missing[:10]}")
    # The authors' MATLAB generator clips requested ROIs at the right/bottom edge.
    out["requested_width"] = out["width"]
    out["requested_height"] = out["height"]
    out["width"] = np.minimum(out["width"], out["parent_width"] - out["x"])
    out["height"] = np.minimum(out["height"], out["parent_height"] - out["y"])
    return out
