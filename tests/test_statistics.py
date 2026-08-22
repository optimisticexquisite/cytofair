import numpy as np
import pandas as pd

from cytofair.statistics import (
    apply_bh_family,
    mean_sd,
    paired_contrasts,
    shuffled_probe_check,
    verify_protocol_order,
)
from cytofair.verification import acceptance_contract_report


def run_table():
    rows = []
    for architecture in ("a", "b"):
        for seed in (17, 29, 43, 71, 101):
            noise = (seed % 7) / 1000
            for protocol, base in (("P1", 0.90), ("P2", 0.80), ("P3", 0.75)):
                rows.append(
                    {
                        "dataset": "d",
                        "architecture": architecture,
                        "seed": seed,
                        "protocol": protocol,
                        "accuracy": base + noise,
                    }
                )
    return pd.DataFrame(rows)


def test_summary_requires_five_runs():
    frame = run_table()
    summary = mean_sd(frame, value="accuracy", by=["architecture", "protocol"])
    assert summary.complete.all()
    short = mean_sd(frame[frame.seed != 101], value="accuracy", by=["architecture", "protocol"])
    assert not short.complete.any()
    assert short["mean"].isna().all()


def test_paired_inference_emits_effect_and_bh_q():
    contrasts = paired_contrasts(
        run_table(),
        value="accuracy",
        family_columns=["dataset", "architecture"],
        left_protocol="P1",
        right_protocol="P2",
        bootstrap_repetitions=100,
    )
    assert len(contrasts) == 2
    assert contrasts.p_value.notna().all()
    assert contrasts.q_value.notna().all()
    assert contrasts.effect_name.ne("").all()
    assert contrasts.effect_size.notna().all()


def test_bh_can_be_applied_over_one_combined_declared_family():
    first = paired_contrasts(
        run_table(),
        value="accuracy",
        family_columns=["dataset", "architecture"],
        left_protocol="P1",
        right_protocol="P2",
        bootstrap_repetitions=20,
        apply_bh=False,
    )
    second = first.copy()
    combined = apply_bh_family(pd.concat([first, second], ignore_index=True))
    assert combined.q_value.notna().all()


def test_expected_protocol_order():
    ordering = verify_protocol_order(
        run_table(), value="accuracy", by=["dataset", "architecture"]
    )
    assert ordering.ordering_passed.all()


def test_shuffled_probe_confirms_chance():
    check = shuffled_probe_check([0.098, 0.101, 0.099, 0.102, 0.100], chance=0.1)
    assert check["passed"]
    with np.testing.assert_raises(ValueError):
        shuffled_probe_check([0.1] * 4, chance=0.1)


def test_shuffled_probe_handles_constant_scores():
    check = shuffled_probe_check([0.1] * 5, chance=0.1)
    assert check["passed"]
    assert check["ci_low"] == check["ci_high"] == 0.1


def test_acceptance_report_never_silently_passes_missing_evidence(tmp_path):
    import subprocess

    subprocess.run(["git", "init", "-q", tmp_path], check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    (tmp_path / "PREREGISTRATION.md").write_text("registered\n")
    subprocess.run(["git", "add", "PREREGISTRATION.md"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "preregister"], cwd=tmp_path, check=True)
    report = acceptance_contract_report(repository=tmp_path)
    assert set(report.status) == {"PENDING"}
