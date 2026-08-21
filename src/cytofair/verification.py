"""Cross-cutting release and statistics verification helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd


def git_preregistration_evidence(repository: str | Path) -> dict[str, object]:
    repository = Path(repository)

    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=repository, text=True).strip()

    prereg_log = git(
        "log", "--follow", "--diff-filter=A", "--format=%H%x09%aI%x09%s", "--", "PREREGISTRATION.md"
    )
    prereg_line = prereg_log.splitlines()[-1] if prereg_log else ""
    prereg_parts = prereg_line.split("\t", 2) if prereg_line else []
    result_log = git(
        "log",
        "--reverse",
        "--diff-filter=A",
        "--format=%H%x09%aI%x09%s",
        "--",
        "results",
        "reports/generated",
    )
    result_line = result_log.splitlines()[0] if result_log else ""
    result_parts = result_line.split("\t", 2) if result_line else []
    prereg_sha = prereg_parts[0] if prereg_parts else None
    result_sha = result_parts[0] if result_parts else None
    if prereg_sha and result_sha:
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", prereg_sha, result_sha],
            cwd=repository,
            check=False,
        ).returncode == 0
    else:
        ancestor = None
    return {
        "preregistration_commit": prereg_sha,
        "preregistration_date": prereg_parts[1] if len(prereg_parts) > 1 else None,
        "first_result_commit": result_sha,
        "first_result_date": result_parts[1] if len(result_parts) > 1 else None,
        "preregistration_predates_results": ancestor,
        "status": "pending_first_result" if result_sha is None else ("passed" if ancestor else "failed"),
    }


def write_markdown_table(frame: pd.DataFrame, path: str | Path, title: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n{frame.to_markdown(index=False)}\n")
