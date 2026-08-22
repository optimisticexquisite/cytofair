#!/usr/bin/env python3
"""Build the Section-11 PASS/FAIL/PENDING ledger from persisted evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from cytofair.verification import acceptance_contract_report


def _csv(path: str) -> pd.DataFrame | None:
    source = Path(path)
    return pd.read_csv(source) if source.is_file() else None


def _json(path: str) -> dict[str, object] | None:
    source = Path(path)
    return json.loads(source.read_text()) if source.is_file() else None


def _markdown_table(frame: pd.DataFrame) -> str:
    values = frame.fillna("").astype(str).map(lambda value: value.replace("|", "\\|"))
    header = "| " + " | ".join(values.columns) + " |"
    divider = "| " + " | ".join("---" for _ in values.columns) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in values.itertuples(index=False, name=None)]
    return "\n".join([header, divider, *rows])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", default=".")
    parser.add_argument("--table1", default="results/table1_structural_leakage.csv")
    parser.add_argument(
        "--provenance-validation", default="reports/generated/sipakmed_validation.csv"
    )
    parser.add_argument(
        "--probe-control", default="results/parent_probe_shuffled_control.json"
    )
    parser.add_argument(
        "--classification-summary", default="results/classification_summary.csv"
    )
    parser.add_argument(
        "--primary-contrasts", default="results/classification_primary_contrasts.csv"
    )
    parser.add_argument(
        "--ordering", default="results/classification_protocol_ordering.csv"
    )
    parser.add_argument(
        "--classification-validation",
        default="reports/generated/classification_matrix_validation.json",
    )
    parser.add_argument(
        "--clean-clone-evidence", default="reports/generated/clean_clone_evidence.json"
    )
    parser.add_argument("--paper-checklist", default="reports/paper_checklist.json")
    parser.add_argument("--output", default="reports/generated/acceptance_contract.csv")
    parser.add_argument(
        "--markdown", default="reports/generated/acceptance_contract.md"
    )
    args = parser.parse_args()

    report = acceptance_contract_report(
        repository=args.repository,
        table1=_csv(args.table1),
        provenance_validation=_csv(args.provenance_validation),
        probe_control=_json(args.probe_control),
        classification_summary=_csv(args.classification_summary),
        primary_contrasts=_csv(args.primary_contrasts),
        classification_ordering=_csv(args.ordering),
        classification_validation=_json(args.classification_validation),
        clean_clone_evidence=_json(args.clean_clone_evidence),
        paper_checklist=_json(args.paper_checklist),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output, index=False)
    markdown = Path(args.markdown)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("# Section 11 acceptance contract\n\n" + _markdown_table(report) + "\n")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
