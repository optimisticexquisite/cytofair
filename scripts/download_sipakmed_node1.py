#!/usr/bin/env python3
# ruff: noqa
"""Python-3.6-compatible bounded downloader for node1's intermittent connection.

This operational helper deliberately uses at most three HTTP connections. It resumes
partial segments after socket timeouts and downloads the five archives sequentially.
The package's regular `cytofair download` command remains the release entry point.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import os
import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def remote_size(url):
    request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(request, timeout=180) as response:
        content_range = response.headers.get("Content-Range")
        if content_range:
            return int(content_range.rsplit("/", 1)[1])
        return int(response.headers["Content-Length"])


def segment(url, path, start, end):
    expected = end - start + 1
    for attempt in range(20):
        current = os.path.getsize(path) if os.path.exists(path) else 0
        if current == expected:
            return
        if current > expected:
            raise IOError("Oversized partial segment: {}".format(path))
        request = urllib.request.Request(
            url, headers={"Range": "bytes={}-{}".format(start + current, end)}
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                if response.status != 206:
                    raise IOError("Expected HTTP 206, got {}".format(response.status))
                with open(path, "ab") as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
        except Exception as error:
            print("retry {} attempt {}: {}".format(os.path.basename(path), attempt + 1, error))
            time.sleep(min(60, 5 * (attempt + 1)))
    current = os.path.getsize(path) if os.path.exists(path) else 0
    if current != expected:
        raise IOError("Incomplete segment {}: {} != {}".format(path, current, expected))


def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            value.update(chunk)
    return value.hexdigest()


def download(item, output_root, connections):
    output = os.path.join(output_root, item["filename"])
    size = item.get("size") or remote_size(item["url"])
    if os.path.exists(output) and os.path.getsize(output) == size:
        print("complete {} {} {}".format(output, size, digest(output)))
        return
    part_dir = output + ".parts"
    if not os.path.isdir(part_dir):
        os.makedirs(part_dir)
    span = (size + connections - 1) // connections
    jobs = []
    for index in range(connections):
        start = index * span
        end = min(size - 1, start + span - 1)
        if start <= end:
            jobs.append((item["url"], os.path.join(part_dir, "%03d.part" % index), start, end))
    print("downloading {} bytes={} connections={}".format(item["filename"], size, len(jobs)))
    with ThreadPoolExecutor(max_workers=connections) as pool:
        list(pool.map(lambda args: segment(*args), jobs))
    temporary = output + ".assembling"
    with open(temporary, "wb") as destination:
        for _, path, _, _ in jobs:
            with open(path, "rb") as source:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
    if os.path.getsize(temporary) != size:
        raise IOError("Assembled size mismatch for {}".format(output))
    os.rename(temporary, output)
    shutil.rmtree(part_dir)
    print("complete {} {} {}".format(output, size, digest(output)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("output")
    parser.add_argument("--connections", type=int, default=3, choices=(1, 2, 3))
    args = parser.parse_args()
    with open(args.manifest) as source:
        manifest = json.load(source)
    items = [item for item in manifest["sources"] if item["dataset"] == "sipakmed"]
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    for item in items:
        download(item, args.output, args.connections)


if __name__ == "__main__":
    main()
