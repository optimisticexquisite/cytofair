import importlib.util
from pathlib import Path

import pandas as pd


def _module():
    path = Path(__file__).parents[1] / "scripts" / "generate_paper_tables.py"
    spec = importlib.util.spec_from_file_location("generate_paper_tables", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_classification_table_rejects_bare_p_value():
    module = _module()
    summary_rows = []
    for protocol in module.PROTOCOL_LABEL:
        summary_rows.append(
            {
                "dataset": "d",
                "architecture": "resnet18",
                "protocol": protocol,
                "metric": "accuracy",
                "n": 5,
                "mean_sd": "0.8000 ± 0.0100",
                "complete": True,
            }
        )
    contrast = pd.DataFrame(
        [
            {
                "dataset": "d",
                "architecture": "resnet18",
                "left_protocol": "P1",
                "right_protocol": "P2",
                "n": 5,
                "mean_difference": 0.1,
                "sd_difference": 0.01,
                "test": "paired_t",
                "p_value": 0.001,
                "q_value": 0.002,
                "effect_name": "cohens_dz",
                "effect_size": 2.0,
                "effect_ci_low": 1.0,
                "effect_ci_high": 3.0,
                "multiplicity_family": "one_family",
            }
        ]
    )
    scores, inference = module.classification_tables(pd.DataFrame(summary_rows), contrast)
    assert "0.8000 ± 0.0100" in scores
    assert "p=0.0010, q=0.0020; cohens_dz=2.000 [1.000, 3.000]" in inference
    contrast.loc[0, "effect_size"] = float("nan")
    try:
        module.classification_tables(pd.DataFrame(summary_rows), contrast)
    except ValueError as error:
        assert "Every p-value" in str(error)
    else:
        raise AssertionError("A bare p-value was accepted")


def test_secondary_and_ordering_tables_make_violations_explicit():
    module = _module()
    secondary = pd.DataFrame(
        [
            {
                "dataset": "d",
                "architecture": "resnet18",
                "left_protocol": "P2",
                "right_protocol": "P3",
                "n": 5,
                "mean_difference": -0.01,
                "sd_difference": 0.02,
                "test": "paired_t",
                "p_value": 0.4,
                "effect_name": "cohens_dz",
                "effect_size": -0.5,
                "effect_ci_low": -1.2,
                "effect_ci_high": 0.2,
            }
        ]
    )
    rendered = module.secondary_contrast_table(secondary)
    assert "unadjusted p=0.4000; cohens_dz=-0.500 [-1.200, 0.200]" in rendered
    ordering = module.ordering_table(
        pd.DataFrame(
            [{"dataset": "d", "architecture": "resnet18", "ordering_passed": False}]
        )
    )
    assert "Violation" in ordering


def test_ranking_table_requires_effect_and_ci_for_p_value():
    module = _module()
    ranking = pd.DataFrame(
        [
            {
                "dataset": "d",
                "left_protocol": "P1",
                "right_protocol": "P2",
                "n_architectures": 6,
                "n_seeds": 5,
                "p_value": 0.2,
                "effect_name": "kendall_tau",
                "effect_size": 0.6,
                "effect_ci_low": 0.1,
                "effect_ci_high": 1.0,
                "reordered_pairs": 3,
                "total_pairs": 15,
                "tie_changes": 0,
            }
        ]
    )
    rendered = module.ranking_table(ranking)
    assert "0.600 [0.100, 1.000]" in rendered
    assert "3/15" in rendered
    ranking.loc[0, "effect_ci_low"] = float("nan")
    try:
        module.ranking_table(ranking)
    except ValueError as error:
        assert "95% CI" in str(error)
    else:
        raise AssertionError("Ranking p-value without effect CI was accepted")
