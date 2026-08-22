import importlib.util
from pathlib import Path

import pandas as pd


def test_figure1_states_finding_and_control():
    path = Path(__file__).parents[1] / "scripts" / "generate_figure1.py"
    spec = importlib.util.spec_from_file_location("generate_figure1", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    table = pd.DataFrame(
        {
            "dataset": ["SIPaKMeD", "Herlev"],
            "observed_p1_straddle_pct_mean_sd": ["45.63 ± 1.04", "0.00 ± 0.00"],
        }
    )
    svg = module.figure_svg(table)
    assert "group by parent before splitting" in svg
    assert "Herlev" in svg
    assert "0.00%" in svg
