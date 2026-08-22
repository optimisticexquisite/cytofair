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
