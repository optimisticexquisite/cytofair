"""Checksum-aware, resumable raw-source downloader with bounded concurrency."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def remote_size(url: str, timeout: int = 60) -> int:
    request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content_range = response.headers.get("Content-Range")
        if content_range and "/" in content_range:
            return int(content_range.rsplit("/", 1)[1])
        length = response.headers.get("Content-Length")
        if length is None:
            raise RuntimeError(f"Server did not expose size for {url}")
        return int(length)


def _download_segment(url: str, path: Path, start: int, end: int, timeout: int) -> None:
    expected = end - start + 1
    if path.exists() and path.stat().st_size == expected:
        return
    temporary = path.with_suffix(path.suffix + ".downloading")
    for attempt in range(8):
        current = temporary.stat().st_size if temporary.exists() else 0
        if current == expected:
            break
        if current > expected:
            raise OSError(f"Oversized partial segment {temporary}")
        request = urllib.request.Request(
            url, headers={"Range": f"bytes={start + current}-{end}"}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response, temporary.open(
                "ab"
            ) as dst:
                if response.status != 206:
                    raise RuntimeError(f"Range request returned HTTP {response.status} for {url}")
                shutil.copyfileobj(response, dst, length=1024 * 1024)
        except Exception:
            if attempt == 7:
                raise
            time.sleep(min(30, 2 ** (attempt + 1)))
    if temporary.stat().st_size != expected:
        raise OSError(f"Short segment {temporary}: {temporary.stat().st_size} != {expected}")
    temporary.replace(path)


def download(
    url: str,
    destination: str | Path,
    *,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
    connections: int = 3,
    timeout: int = 180,
) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    size = expected_size or remote_size(url, timeout=timeout)
    if (
        destination.exists()
        and destination.stat().st_size == size
        and (not expected_sha256 or sha256(destination) == expected_sha256)
    ):
        return destination
    part_dir = destination.with_name(destination.name + ".parts")
    part_dir.mkdir(parents=True, exist_ok=True)
    segment_size = (size + connections - 1) // connections
    jobs = []
    for index in range(connections):
        start = index * segment_size
        end = min(size - 1, start + segment_size - 1)
        if start <= end:
            jobs.append((url, part_dir / f"{index:03d}.part", start, end, timeout))
    with ThreadPoolExecutor(max_workers=connections) as pool:
        list(pool.map(lambda args: _download_segment(*args), jobs))
    temporary = destination.with_suffix(destination.suffix + ".assembling")
    with temporary.open("wb") as dst:
        for _, path, _, _, _ in jobs:
            with path.open("rb") as src:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
    if temporary.stat().st_size != size:
        raise OSError(f"Assembled size mismatch for {destination}")
    if expected_sha256 and sha256(temporary) != expected_sha256:
        raise OSError(f"SHA-256 mismatch for {destination}")
    temporary.replace(destination)
    shutil.rmtree(part_dir)
    return destination


def download_manifest(
    manifest_path: str | Path,
    output_root: str | Path,
    *,
    dataset: str | None = None,
    connections: int = 3,
) -> list[Path]:
    manifest = json.loads(Path(manifest_path).read_text())
    selected = [item for item in manifest["sources"] if dataset is None or item["dataset"] == dataset]
    if not selected:
        raise ValueError(f"No sources selected for dataset={dataset!r}")
    outputs = []
    for item in selected:
        outputs.append(
            download(
                item["url"],
                Path(output_root) / item["dataset"] / item["filename"],
                expected_size=item.get("size"),
                expected_sha256=item.get("sha256"),
                connections=connections,
            )
        )
    return outputs
