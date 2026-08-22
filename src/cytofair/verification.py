"""Cross-cutting release and statistics verification helpers."""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .constants import SIPAKMED_PARENT_COUNTS


@dataclass(frozen=True)
class AcceptanceItem:
    section: str
    criterion: str
    status: str
    evidence: str


def _item(section: str, criterion: str, passed: bool | None, evidence: str) -> AcceptanceItem:
    status = "PENDING" if passed is None else ("PASS" if passed else "FAIL")
    return AcceptanceItem(section, criterion, status, evidence)


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


def acceptance_contract_report(
    *,
    repository: str | Path,
    table1: pd.DataFrame | None = None,
    provenance_validation: pd.DataFrame | None = None,
    probe_control: dict[str, object] | None = None,
    classification_summary: pd.DataFrame | None = None,
    primary_contrasts: pd.DataFrame | None = None,
    classification_ordering: pd.DataFrame | None = None,
    classification_validation: dict[str, object] | None = None,
    clean_clone_evidence: dict[str, object] | None = None,
    paper_checklist: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Render every Section-11 gate as PASS, FAIL, or explicitly PENDING."""
    items: list[AcceptanceItem] = []
    if provenance_validation is None:
        items.append(_item("provenance", "all recovery checks", None, "validation absent"))
    else:
        required = {
            "branch_parent_agreement",
            "bbox_within_parent",
            "dataset_total",
            "per_parent_count_sum",
            *(f"parent_count:{name}" for name in SIPAKMED_PARENT_COUNTS),
        }
        available = set(provenance_validation["name"])
        missing = sorted(required - available)
        selected = provenance_validation.loc[provenance_validation.name.isin(required)]
        passed = selected["passed"].astype(bool)
        items.append(
            _item(
                "provenance",
                "A/B agreement, bbox containment, totals, and class counts",
                None if missing else bool(passed.all()),
                f"missing checks: {missing}"
                if missing
                else f"{int(passed.sum())}/{len(passed)} required checks pass",
            )
        )

    if table1 is None:
        for criterion in (
            "analytic straddle agreement",
            "Herlev negative control",
            "P3 <= P2 <= P1",
        ):
            items.append(_item("leakage", criterion, None, "Table 1 absent"))
    else:
        items.append(
            _item(
                "leakage",
                "analytic straddle agreement",
                bool(table1.analytic_check_passed.astype(bool).all()),
                f"{int(table1.analytic_check_passed.astype(bool).sum())}/{len(table1)} datasets",
            )
        )
        control = table1.loc[table1.dataset.str.lower() == "herlev"]
        items.append(
            _item(
                "leakage",
                "Herlev negative control",
                bool(len(control) == 1 and control.negative_control_passed.astype(bool).all()),
                "Herlev row present and exactly zero" if len(control) == 1 else "Herlev row absent",
            )
        )
        items.append(
            _item(
                "leakage",
                "P3 <= P2 <= P1",
                bool(table1.protocol_ordering_passed.astype(bool).all()),
                f"{int(table1.protocol_ordering_passed.astype(bool).sum())}/{len(table1)} datasets",
            )
        )
    items.append(
        _item(
            "leakage",
            "label-shuffled probe covers chance",
            None if probe_control is None else bool(probe_control.get("passed")),
            "probe control absent" if probe_control is None else str(probe_control),
        )
    )

    if classification_summary is None:
        items.append(_item("statistics", "mean ± SD over >=5 seeds", None, "summary absent"))
    else:
        complete = classification_summary.complete.astype(bool)
        items.append(
            _item(
                "statistics",
                "mean ± SD over >=5 seeds",
                bool(complete.all()),
                f"{int(complete.sum())}/{len(complete)} cells complete",
            )
        )
    if primary_contrasts is None:
        items.append(
            _item("statistics", "BH family and effect for every p-value", None, "contrasts absent")
        )
    else:
        reported = primary_contrasts.p_value.notna()
        valid = (
            primary_contrasts.loc[reported, "q_value"].notna()
            & primary_contrasts.loc[reported, "effect_size"].notna()
            & primary_contrasts.loc[reported, "effect_name"].astype(str).ne("")
        )
        one_family = primary_contrasts.multiplicity_family.nunique() == 1
        items.append(
            _item(
                "statistics",
                "BH family and effect for every p-value",
                bool(valid.all() and one_family),
                f"{int(valid.sum())}/{int(reported.sum())} p-values; one_family={one_family}",
            )
        )
    if classification_ordering is not None:
        items.append(
            _item(
                "leakage",
                "model-result P3 <= P2 <= P1",
                bool(classification_ordering.ordering_passed.astype(bool).all()),
                f"{int(classification_ordering.ordering_passed.astype(bool).sum())}/"
                f"{len(classification_ordering)} cells",
            )
        )
    items.append(
        _item(
            "statistics",
            "classification matrix keys, runtime, and split hashes",
            None
            if classification_validation is None
            else bool(classification_validation.get("passed")),
            "classification matrix validation absent"
            if classification_validation is None
            else str(classification_validation),
        )
    )

    git_evidence = git_preregistration_evidence(repository)
    items.append(
        _item(
            "statistics",
            "preregistration predates first result commit",
            git_evidence["preregistration_predates_results"],
            str(git_evidence),
        )
    )
    items.append(
        _item(
            "release",
            "clean clone reproduces Table 1",
            None if clean_clone_evidence is None else bool(clean_clone_evidence.get("passed")),
            "clean-clone evidence absent" if clean_clone_evidence is None else str(clean_clone_evidence),
        )
    )
    paper_keys = {
        "title_adjectives_have_rows": "title adjectives have corresponding results rows",
        "size_matched_in_main_text": "size-matched control is in main text",
        "no_paper_named_deficient": "no individual paper is named deficient",
        "own_affected_papers_listed": "lab's affected papers are included",
        "figure1_reader_test_passed": "Fig. 1 reader states finding correctly",
    }
    for key, criterion in paper_keys.items():
        value = None if paper_checklist is None or key not in paper_checklist else bool(
            paper_checklist[key]
        )
        items.append(
            _item(
                "paper",
                criterion,
                value,
                "checklist evidence absent" if value is None else f"{key}={value}",
            )
        )
    return pd.DataFrame(asdict(item) for item in items)
