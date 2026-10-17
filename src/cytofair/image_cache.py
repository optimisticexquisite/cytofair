"""Packed, checksummed uint8 crop caches for node-local training input.

The network home on the training host is too slow to feed GPUs directly, so each
released crop is decoded once and concatenated into a single flat ``uint8`` array
that every training process memory-maps from node-local storage. The index keeps
one row per crop with its byte offset and shape, and the manifest records the
SHA-256 of both files so a run can prove which pixels it consumed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

CACHE_VERSION = 1


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_cache(
    items: Iterable[tuple[dict, np.ndarray]],
    output_dir: str | Path,
    name: str,
    *,
    source: dict,
) -> dict:
    """Write ``(row, HxWx3 uint8 array)`` items as ``{name}_pixels.npy`` plus index."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, chunks, offset = [], [], 0
    for row, pixels in items:
        pixels = np.ascontiguousarray(pixels, dtype=np.uint8)
        if pixels.ndim != 3 or pixels.shape[2] != 3:
            raise ValueError(f"{row.get('crop_id')}: expected HxWx3 pixels, got {pixels.shape}")
        payload = pixels.tobytes()
        rows.append(
            {
                **row,
                "offset": offset,
                "height": int(pixels.shape[0]),
                "width": int(pixels.shape[1]),
                "pixel_sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
        chunks.append(pixels.reshape(-1))
        offset += pixels.size
    if not rows:
        raise ValueError("Refusing to write an empty crop cache")
    index = pd.DataFrame(rows)
    if not index.crop_id.is_unique:
        raise ValueError("Crop cache IDs must be unique")
    pixels_path = output_dir / f"{name}_pixels.npy"
    index_path = output_dir / f"{name}_index.parquet"
    manifest_path = output_dir / f"{name}_manifest.json"
    temporary_pixels = pixels_path.with_name(pixels_path.stem + ".tmp.npy")
    np.save(temporary_pixels, np.concatenate(chunks))
    temporary_pixels.replace(pixels_path)
    temporary_index = index_path.with_name(index_path.stem + ".tmp.parquet")
    index.to_parquet(temporary_index, index=False)
    temporary_index.replace(index_path)
    manifest = {
        "cache_version": CACHE_VERSION,
        "name": name,
        "n_crops": len(index),
        "total_pixel_bytes": int(offset),
        "pixels_sha256": _sha256_file(pixels_path),
        "index_sha256": _sha256_file(index_path),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source": source,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def build_file_cache(
    provenance: pd.DataFrame,
    output_dir: str | Path,
    name: str,
    *,
    path_column: str = "crop_path",
    keep_columns: tuple[str, ...] = ("parent_id", "class_name"),
) -> dict:
    """Decode every released crop file exactly as the Study 1 loader did (PIL, RGB)."""
    required = {"crop_id", path_column, *keep_columns}
    missing = sorted(required - set(provenance.columns))
    if missing:
        raise ValueError(f"File cache provenance misses: {missing}")

    def items() -> Iterator[tuple[dict, np.ndarray]]:
        for row in provenance.sort_values("crop_id").itertuples(index=False):
            with Image.open(getattr(row, path_column)) as image:
                pixels = np.asarray(image.convert("RGB"))
            yield {"crop_id": row.crop_id, **{c: getattr(row, c) for c in keep_columns}}, pixels

    return _write_cache(
        items(),
        output_dir,
        name,
        source={"kind": "released_crop_files", "rows": len(provenance)},
    )


def window_origin(center: int, size: int, limit: int) -> int:
    """Top-left coordinate of a ``size`` window centred on ``center`` inside ``limit``.

    Windows that would cross the image border are shifted inwards rather than
    padded, so every crop contains only real source pixels.
    """
    if size > limit:
        raise ValueError(f"window {size} exceeds image extent {limit}")
    return int(min(max(center - size // 2, 0), limit - size))


def build_window_cache(
    cells: pd.DataFrame,
    image_root: str | Path,
    output_dir: str | Path,
    name: str,
    *,
    size: int = 100,
    keep_columns: tuple[str, ...] = ("parent_id", "class_name"),
) -> dict:
    """Cut fixed-size windows centred on published nucleus coordinates (CRIC)."""
    required = {"crop_id", "source_name", "center_x", "center_y", *keep_columns}
    missing = sorted(required - set(cells.columns))
    if missing:
        raise ValueError(f"Window cache cells miss: {missing}")
    image_root = Path(image_root)

    def items() -> Iterator[tuple[dict, np.ndarray]]:
        for source_name, group in cells.sort_values("crop_id").groupby("source_name", sort=True):
            with Image.open(image_root / str(source_name)) as image:
                source = np.asarray(image.convert("RGB"))
            height, width = source.shape[:2]
            for row in group.itertuples(index=False):
                x0 = window_origin(int(row.center_x), size, width)
                y0 = window_origin(int(row.center_y), size, height)
                yield (
                    {
                        "crop_id": row.crop_id,
                        **{c: getattr(row, c) for c in keep_columns},
                        "source_name": source_name,
                        "x": x0,
                        "y": y0,
                        "window": size,
                        "source_width": width,
                        "source_height": height,
                    },
                    source[y0 : y0 + size, x0 : x0 + size],
                )

    return _write_cache(
        items(),
        output_dir,
        name,
        source={"kind": "nucleus_centred_windows", "size": size, "rows": len(cells)},
    )


class CropCache:
    """Read-only view of a packed crop cache; safe to pickle into DataLoader workers."""

    def __init__(self, directory: str | Path, name: str, *, verify: bool = False):
        self.directory = Path(directory)
        self.name = name
        self.manifest = json.loads((self.directory / f"{name}_manifest.json").read_text())
        if self.manifest.get("cache_version") != CACHE_VERSION:
            raise ValueError(f"Unsupported crop cache version: {self.manifest.get('cache_version')}")
        self.index = pd.read_parquet(self.directory / f"{name}_index.parquet")
        if verify:
            self.verify()
        self._lookup = {
            crop_id: (int(offset), int(height), int(width))
            for crop_id, offset, height, width in zip(
                self.index.crop_id, self.index.offset, self.index.height, self.index.width, strict=True
            )
        }
        self._pixels = None

    @property
    def pixels_path(self) -> Path:
        return self.directory / f"{self.name}_pixels.npy"

    def verify(self) -> None:
        observed = {
            "pixels_sha256": _sha256_file(self.pixels_path),
            "index_sha256": _sha256_file(self.directory / f"{self.name}_index.parquet"),
        }
        for key, value in observed.items():
            if value != self.manifest[key]:
                raise ValueError(f"Crop cache {self.name} failed {key} verification")

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_pixels"] = None
        return state

    def __contains__(self, crop_id: str) -> bool:
        return crop_id in self._lookup

    def array(self, crop_id: str) -> np.ndarray:
        if self._pixels is None:
            self._pixels = np.load(self.pixels_path, mmap_mode="r")
        offset, height, width = self._lookup[crop_id]
        return np.asarray(self._pixels[offset : offset + height * width * 3]).reshape(height, width, 3)

    def image(self, crop_id: str) -> Image.Image:
        # An HxWx3 uint8 array is decoded as RGB; Pillow 12 deprecates ``mode=``.
        return Image.fromarray(np.ascontiguousarray(self.array(crop_id)))


def write_array_cache(
    items: Iterable[tuple[dict, np.ndarray]], output_dir: str | Path, name: str, *, source: dict
) -> dict:
    """Public wrapper used for derived image sets such as counterfactual composites."""
    return _write_cache(items, output_dir, name, source=source)
