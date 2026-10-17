#!/usr/bin/env python3
"""List the 400 CRIC Cervix slide images on Figshare with their download URLs and MD5s.

The Figshare API is queried once per image article; the output manifest is later
used to download every image with checksum verification (``fetch_cric_images.py``).
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request

COLLECTION = 4960286


def get_json(url: str, attempts: int = 6):
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.load(response)
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    args = parser.parse_args()
    articles = get_json(f"https://api.figshare.com/v2/collections/{COLLECTION}/articles?page_size=1000")
    images = [a for a in articles if a["title"].startswith("CRIC Cervix Microscope Slide Image #")]
    records = []
    for article in sorted(images, key=lambda a: int(a["title"].rsplit("#", 1)[1])):
        detail = get_json(article["url_public_api"])
        files = detail["files"]
        if len(files) != 1:
            raise RuntimeError(f"{article['title']}: expected one file, found {len(files)}")
        file = files[0]
        records.append(
            {
                "image_number": int(article["title"].rsplit("#", 1)[1]),
                "article_id": article["id"],
                "doi": detail["doi"],
                "license": detail["license"]["name"],
                "filename": file["name"],
                "url": file["download_url"],
                "size": file["size"],
                "md5": file["computed_md5"],
            }
        )
        time.sleep(0.25)
    if len(records) != 400:
        raise RuntimeError(f"Expected 400 CRIC images, listed {len(records)}")
    with open(args.output, "w") as handle:
        json.dump({"collection": COLLECTION, "images": records}, handle, indent=2)
        handle.write("\n")
    print(f"listed {len(records)} images, {sum(r['size'] for r in records) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
