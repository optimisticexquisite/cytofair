#!/usr/bin/env python3
"""Print timing and artifact completeness for run JSONs (operational check only)."""

import json
import sys
from pathlib import Path

for path in sorted(Path(sys.argv[1]).glob("*.json")):
    record = json.loads(path.read_text())
    history = record["history"]
    finite = all(item["train_loss"] == item["train_loss"] for item in history)
    rate = record["train_crops"] * len(history) / record["training_seconds"]
    print(
        f"{record['architecture']:16s} {record['protocol']:10s} epochs={len(history):2d} "
        f"train_s={record['training_seconds']:7.1f} total_s={record['elapsed_seconds']:7.1f} "
        f"img/s={rate:7.1f} finite={finite} artifacts={sorted(record['artifacts'])}"
    )
