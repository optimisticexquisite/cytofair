"""Materialize the released Cx22 MATLAB labels and generator-defined RGB ROIs."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image

ASSET_SCHEMA_VERSION = 2
TRAINING_CACHE_VERSION = 1


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
    expected_instances_per_target: int = 7473,
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

    version_path = output_root / "asset_schema.json"
    expected_version = {"asset_schema_version": ASSET_SCHEMA_VERSION}
    try:
        current_version = json.loads(version_path.read_text())
    except (OSError, ValueError):
        current_version = None
    overwrite = overwrite or current_version != expected_version

    rows = []
    ordered = provenance.sort_values("crop_id").reset_index(drop=True)
    for subset in sorted(provenance.subset.unique()):
        directory = _subset_directory(label_root, str(subset))
        cyto = read_instance_masks(directory / "cyto" / "cyto_ins.mat", "cyto_ins")
        nuc = read_instance_masks(directory / "nuc" / "nuc_ins.mat", "nuc_ins")
        subset_rows = ordered.loc[ordered.subset == subset]
        expected_subset = len(subset_rows)
        if len(cyto) != expected_subset or len(nuc) != expected_subset:
            raise ValueError(
                f"{subset} label count mismatch: provenance={expected_subset}, "
                f"cytoplasm={len(cyto)}, nucleus={len(nuc)}"
            )
        for row in subset_rows.itertuples(index=False):
            index = int(str(row.crop_id).rsplit("/", 1)[1])
            cyto_instances, nuc_instances = cyto[index], nuc[index]
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
                        raise ValueError(
                            f"Materialized Cx22 image has wrong size: {path} {image.size}"
                        )
            rows.append(
                {
                    **row._asdict(),
                    "asset_schema_version": ASSET_SCHEMA_VERSION,
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
        # Bound peak memory to one release subset rather than all 1,320 ROIs.
        del cyto, nuc
    manifest = pd.DataFrame(rows)
    if len(manifest) != expected_total or manifest[
        ["crop_path", "cyto_mask_path", "nuc_mask_path", "cyto_instances_path"]
    ].isna().any().any():
        raise RuntimeError("Cx22 materialization failed its final completeness gate")
    observed_instances = (
        int(manifest.n_cyto_instances.sum()),
        int(manifest.n_nuc_instances.sum()),
    )
    expected_instances = (expected_instances_per_target, expected_instances_per_target)
    if observed_instances != expected_instances:
        raise ValueError(
            f"Cx22 instance-count gate failed: {observed_instances} != {expected_instances}"
        )
    version_path.write_text(json.dumps(expected_version, indent=2) + "\n")
    return manifest


def build_training_cache(
    manifest: pd.DataFrame,
    output_root: str | Path,
    *,
    image_size: int = 256,
    expected_total: int = 1320,
) -> pd.DataFrame:
    """Create contiguous uint8 arrays to avoid repeated network-filesystem decoding."""
    required = {
        "crop_id",
        "crop_path",
        "cyto_mask_path",
        "nuc_mask_path",
        "cyto_instances_path",
        "n_cyto_instances",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Cx22 cache construction misses manifest columns: {missing}")
    if len(manifest) != expected_total or not manifest.crop_id.is_unique:
        raise ValueError(f"Cx22 cache requires {expected_total} unique rows; got {len(manifest)}")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    final_paths = {
        "images": output_root / f"images_{image_size}.npy",
        "semantic": output_root / f"semantic_{image_size}.npy",
        "instances": output_root / f"instances_{image_size}.npy",
    }
    temporary_paths = {
        key: path.with_name(path.stem + ".tmp.npy") for key, path in final_paths.items()
    }
    total_instances = int(manifest.n_cyto_instances.sum())
    arrays = {
        "images": np.lib.format.open_memmap(
            temporary_paths["images"],
            mode="w+",
            dtype=np.uint8,
            shape=(len(manifest), image_size, image_size, 3),
        ),
        "semantic": np.lib.format.open_memmap(
            temporary_paths["semantic"],
            mode="w+",
            dtype=np.uint8,
            shape=(len(manifest), 2, image_size, image_size),
        ),
        "instances": np.lib.format.open_memmap(
            temporary_paths["instances"],
            mode="w+",
            dtype=np.uint8,
            shape=(total_instances, image_size, image_size),
        ),
    }
    rows, instance_offset = [], 0
    for cache_index, row in enumerate(
        manifest.sort_values("crop_id").itertuples(index=False)
    ):
        with Image.open(row.crop_path) as image:
            arrays["images"][cache_index] = np.asarray(
                image.convert("RGB").resize(
                    (image_size, image_size), Image.Resampling.BILINEAR
                )
            )
        for channel, path in enumerate((row.cyto_mask_path, row.nuc_mask_path)):
            with Image.open(path) as mask:
                arrays["semantic"][cache_index, channel] = (
                    np.asarray(
                        mask.convert("L").resize(
                            (image_size, image_size), Image.Resampling.NEAREST
                        )
                    )
                    > 0
                )
        with np.load(row.cyto_instances_path) as archive:
            instances = archive["masks"]
        count = len(instances)
        if count != int(row.n_cyto_instances):
            raise ValueError(f"Instance count changed for {row.crop_id}: {count}")
        for relative_index, mask in enumerate(instances):
            arrays["instances"][instance_offset + relative_index] = (
                np.asarray(
                    Image.fromarray(mask, mode="L").resize(
                        (image_size, image_size), Image.Resampling.NEAREST
                    )
                )
                > 0
            )
        rows.append(
            {
                **row._asdict(),
                "training_cache_version": TRAINING_CACHE_VERSION,
                "cache_image_size": image_size,
                "cache_index": cache_index,
                "cache_images_path": str(final_paths["images"].resolve()),
                "cache_semantic_path": str(final_paths["semantic"].resolve()),
                "cache_instances_path": str(final_paths["instances"].resolve()),
                "cache_instance_start": instance_offset,
                "cache_instance_count": count,
            }
        )
        instance_offset += count
    if instance_offset != total_instances:
        raise RuntimeError(f"Cx22 cache instance count {instance_offset} != {total_instances}")
    for array in arrays.values():
        array.flush()
    del arrays
    for key, temporary in temporary_paths.items():
        temporary.replace(final_paths[key])
    hashes = {f"cache_{key}_sha256": _sha256(path) for key, path in final_paths.items()}
    out = pd.DataFrame(rows)
    for column, value in hashes.items():
        out[column] = value
    marker = {
        "training_cache_version": TRAINING_CACHE_VERSION,
        "image_size": image_size,
        "crops": len(out),
        "cytoplasm_instances": total_instances,
        **hashes,
    }
    (output_root / "cache_manifest.json").write_text(json.dumps(marker, indent=2) + "\n")
    return out
