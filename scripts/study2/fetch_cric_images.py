#!/usr/bin/env python3
"""Download the CRIC slide images listed by ``list_cric_images.py`` with MD5 checks.

At most three downloads run concurrently. Each file is written to a temporary
name, verified against Figshare's MD5 and byte size, and only then renamed. The
SHA-256 of every verified file is appended to the output manifest so later
reproductions can verify the exact bytes that were cached.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def fetch(record: dict, root: Path) -> dict:
    target = root / record["filename"]
    for attempt in range(8):
        if target.is_file() and target.stat().st_size == record["size"]:
            payload = target.read_bytes()
            if hashlib.md5(payload).hexdigest() == record["md5"]:
                return {**record, "sha256": hashlib.sha256(payload).hexdigest()}
        temporary = target.with_suffix(target.suffix + ".part")
        try:
            with urllib.request.urlopen(record["url"], timeout=120) as response:
                temporary.write_bytes(response.read())
            temporary.replace(target)
        except (OSError, http.client.HTTPException):
            temporary.unlink(missing_ok=True)
            time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"Could not verify {record['filename']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("listing")
    parser.add_argument("root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--connections", type=int, default=3, choices=(1, 2, 3))
    args = parser.parse_args()
    listing = json.loads(Path(args.listing).read_text())
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.connections) as pool:
        verified = list(pool.map(lambda record: fetch(record, root), listing["images"]))
    Path(args.output).write_text(json.dumps({**listing, "images": verified}, indent=2) + "\n")
    print(f"verified {len(verified)} CRIC images")


if __name__ == "__main__":
    main()
