"""Deterministic extraction of declared dataset archives."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import py7zr


def _extract_archive(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    sentinel = destination / f".{archive.name}.complete"
    if sentinel.exists():
        return
    if archive.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive) as source:
            source.extractall(destination)
    elif archive.suffix.lower() == ".7z":
        with py7zr.SevenZipFile(archive, mode="r") as source:
            source.extractall(destination)
    else:
        raise ValueError(f"Unsupported archive format: {archive}")
    sentinel.write_text("complete\n")


def extract_manifest(
    manifest_path: str | Path,
    raw_root: str | Path,
    output_root: str | Path,
    *,
    dataset: str | None = None,
) -> list[Path]:
    manifest = json.loads(Path(manifest_path).read_text())
    raw_root, output_root = Path(raw_root), Path(output_root)
    outputs = []
    for item in manifest["sources"]:
        if dataset is not None and item["dataset"] != dataset:
            continue
        archive = raw_root / item["dataset"] / item["filename"]
        if archive.suffix.lower() not in {".zip", ".7z"}:
            continue
        if item["dataset"] == "cx22":
            destination = output_root / item["dataset"] / archive.stem
        else:
            destination = output_root / item["dataset"]
        _extract_archive(archive, destination)
        for nested in item.get("nested_archives", []):
            nested_archive = destination / nested["path"]
            if not nested_archive.is_file():
                raise FileNotFoundError(f"Declared nested archive is absent: {nested_archive}")
            nested_destination = destination / nested["destination"]
            _extract_archive(nested_archive, nested_destination)
        outputs.append(destination)
    return outputs
