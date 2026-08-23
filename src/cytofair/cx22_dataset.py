"""Materialize the released Cx22 MATLAB labels and generator-defined RGB ROIs."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image


def _matlab_round(value: float) -> int:
    """MATLAB rounds positive half-integers away from zero."""
    return math.floor(value + 0.5)


def reconstruct_roi(row: pd.Series, size: int = 512) -> Image.Image:
    """Reproduce ImageDataGenerator.m for one already validated provenance row."""
    requested_width = int(row.get("requested_width", row["width"]))
    requested_height = int(row.get("requested_height", row["height"]))
    x, y = int(row["x"]), int(row["y"])
    with Image.open(row["parent_path"]) as source:
        source = source.convert("RGB")
        right = min(x + requested_width, source.width)
        bottom = min(y + requested_height, source.height)
        roi = source.crop((x, y, right, bottom))
    if requested_height >= size or requested_width >= size:
        scale = size / max(requested_height, requested_width)
        if requested_height >= requested_width:
            resized = (max(1, _matlab_round(scale * roi.width)), size)
        else:
            resized = (size, max(1, _matlab_round(scale * roi.height)))
        roi = roi.resize(resized, Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(roi, (0, 0))
    return canvas


def read_instance_masks(path: str | Path, key: str) -> list[np.ndarray]:
    """Decode a MATLAB-v7.3 outer ROI cell and its inner instance-mask cells."""
    masks: list[np.ndarray] = []
    with h5py.File(path, "r") as handle:
        outer = handle[key]
        for roi_reference in np.asarray(outer).ravel(order="F"):
            inner = handle[roi_reference]
            roi_masks = []
            for mask_reference in np.asarray(inner).ravel(order="F"):
                # MATLAB dimensions are exposed in reverse order by HDF5. The
                # released masks are square, so the transpose is still necessary
                # to preserve their x/y relation to the reconstructed RGB image.
                mask = np.asarray(handle[mask_reference], dtype=np.uint8).T
                if mask.ndim != 2 or mask.shape != (512, 512):
                    raise ValueError(f"Unexpected Cx22 mask shape in {path}:{key}: {mask.shape}")
                if not np.isin(mask, (0, 1)).all():
                    raise ValueError(f"Non-binary Cx22 mask in {path}:{key}")
                roi_masks.append(mask)
            if not roi_masks:
                raise ValueError(f"Empty instance cell in {path}:{key}")
            masks.append(np.stack(roi_masks, axis=0))
    return masks


def _subset_directory(root: Path, subset: str) -> Path:
    candidates = sorted(
        path.parent.parent for path in root.rglob("ImageDataNames.mat") if path.parent.parent.name == subset
    )
    if len(candidates) != 1:
        raise ValueError(f"Expected one label directory for {subset}; found {candidates}")
    return candidates[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def materialize_dataset(
    provenance: pd.DataFrame,
    label_root: str | Path,
    output_root: str | Path,
    *,
    expected_total: int = 1320,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Write RGB, semantic masks, and compressed cytoplasm instances for Cx22."""
    required = {
        "crop_id",
        "parent_id",
        "subset",
        "x",
        "y",
        "width",
        "height",
        "parent_path",
    }
    missing = sorted(required - set(provenance.columns))
    if missing:
        raise ValueError(f"Cx22 materialization misses provenance columns: {missing}")
    if len(provenance) != expected_total or not provenance.crop_id.is_unique:
        raise ValueError(
            f"Cx22 materialization requires {expected_total} unique rows; got {len(provenance)}"
        )
    label_root, output_root = Path(label_root), Path(output_root)
    directories = {
        name: output_root / name for name in ("images", "cytoplasm", "nucleus", "instances")
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    label_sets: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {}
    for subset in sorted(provenance.subset.unique()):
        directory = _subset_directory(label_root, str(subset))
        cyto = read_instance_masks(directory / "cyto" / "cyto_clumps.mat", "cyto_clumps")
        nuc = read_instance_masks(directory / "nuc" / "nuc_clumps.mat", "nuc_clumps")
        expected_subset = int((provenance.subset == subset).sum())
        if len(cyto) != expected_subset or len(nuc) != expected_subset:
            raise ValueError(
                f"{subset} label count mismatch: provenance={expected_subset}, "
                f"cytoplasm={len(cyto)}, nucleus={len(nuc)}"
            )
        label_sets[str(subset)] = (cyto, nuc)

    rows = []
    ordered = provenance.sort_values("crop_id").reset_index(drop=True)
    for row in ordered.itertuples(index=False):
        subset = str(row.subset)
        index = int(str(row.crop_id).rsplit("/", 1)[1])
        cyto_instances, nuc_instances = (values[index] for values in label_sets[subset])
        cyto_union = np.any(cyto_instances, axis=0).astype(np.uint8) * 255
        nuc_union = np.any(nuc_instances, axis=0).astype(np.uint8) * 255
        if not cyto_union.any() or not nuc_union.any():
            raise ValueError(f"{row.crop_id} has an empty released target")
        safe = str(row.crop_id).replace("/", "__")
        image_path = directories["images"] / f"{safe}.png"
        cyto_path = directories["cytoplasm"] / f"{safe}.png"
        nuc_path = directories["nucleus"] / f"{safe}.png"
        instances_path = directories["instances"] / f"{safe}.npz"
        targets = (image_path, cyto_path, nuc_path, instances_path)
        if overwrite or not all(path.is_file() for path in targets):
            reconstruct_roi(pd.Series(row._asdict())).save(image_path, compress_level=6)
            Image.fromarray(cyto_union, mode="L").save(cyto_path, compress_level=6)
            Image.fromarray(nuc_union, mode="L").save(nuc_path, compress_level=6)
            np.savez_compressed(instances_path, masks=cyto_instances.astype(np.uint8))
        for path in targets[:3]:
            with Image.open(path) as image:
                if image.size != (512, 512):
                    raise ValueError(f"Materialized Cx22 image has wrong size: {path} {image.size}")
        rows.append(
            {
                **row._asdict(),
                "crop_path": str(image_path.resolve()),
                "cyto_mask_path": str(cyto_path.resolve()),
                "nuc_mask_path": str(nuc_path.resolve()),
                "cyto_instances_path": str(instances_path.resolve()),
                "n_cyto_instances": len(cyto_instances),
                "n_nuc_instances": len(nuc_instances),
                "image_sha256": _sha256(image_path),
                "cyto_mask_sha256": _sha256(cyto_path),
                "nuc_mask_sha256": _sha256(nuc_path),
                "cyto_instances_sha256": _sha256(instances_path),
            }
        )
    manifest = pd.DataFrame(rows)
    if len(manifest) != expected_total or manifest[
        ["crop_path", "cyto_mask_path", "nuc_mask_path", "cyto_instances_path"]
    ].isna().any().any():
        raise RuntimeError("Cx22 materialization failed its final completeness gate")
    return manifest
