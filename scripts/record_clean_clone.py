#!/usr/bin/env python3
"""Record a successful raw-to-Table-1 run without self-certifying host virginity."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", default="results/table1_structural_leakage.csv")
    parser.add_argument("--output", default="reports/generated/clean_clone_evidence.json")
    parser.add_argument("--virgin-host-attested", action="store_true")
    args = parser.parse_args()
    table = Path(args.table)
    if not table.is_file():
        raise FileNotFoundError(table)
    payload = {
        "passed": bool(args.virgin_host_attested),
        "raw_to_table_command_completed": True,
        "virgin_host_attested": bool(args.virgin_host_attested),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "table1_sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
        "entry_point": "./scripts/reproduce_table1.sh",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
