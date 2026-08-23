#!/usr/bin/env python3
"""Print a compact health report for a detached Cx22 matrix controller."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output", nargs="?", default="results/cx22_segmentation", help="matrix output directory"
    )
    args = parser.parse_args()
    output = Path(args.output)
    status_path = output / "matrix_status.json"
    if not status_path.is_file():
        raise SystemExit(f"No status file yet: {status_path}")
    status = json.loads(status_path.read_text())
    controller_alive = alive(int(status["controller_pid"]))
    print(
        f"state={status['state']} controller_alive={controller_alive} "
        f"completed={status['completed']}/{status['total']} pending={status['pending']} "
        f"updated={status['updated_at_utc']}"
    )
    print(f"gpus={status['gpus']} epochs_per_cell={status['epochs']}")
    for active in status.get("active", []):
        print(
            f"ACTIVE gpu={active['gpu']} pid={active['pid']} "
            f"{active['task']}:{active['architecture']}:{active['protocol']}:seed{active['seed']} "
            f"epoch={active['epoch']}/{status['epochs']} attempt={active['attempt']}"
        )
        print(f"  log={active['log']}")
    for failure in status.get("failures", []):
        print(
            f"FAILED {failure['task']}:{failure['architecture']}:{failure['protocol']}:"
            f"seed{failure['seed']} rc={failure['return_code']} "
            f"epoch={failure['last_epoch']} attempts={failure['attempts']}"
        )
        print(f"  log={failure['last_log']}")
    if status["state"] == "running" and not controller_alive:
        print("STALE: the controller died; paste this report and the active log tails back to Codex.")
        raise SystemExit(2)
    if status["state"] in {"failed", "interrupted"}:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
