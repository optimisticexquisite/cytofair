import importlib.util
from pathlib import Path

import pandas as pd


def test_figure3_requires_complete_matched_protocol_cells():
    path = Path(__file__).parents[1] / "scripts" / "generate_figure3.py"
    spec = importlib.util.spec_from_file_location("generate_figure3", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    rows = []
    for architecture_index, architecture in enumerate(module.ARCHITECTURE_LABELS):
        for protocol_index, protocol in enumerate(module.PROTOCOLS):
            rows.append(
                {
                    "dataset": "SIPaKMeD",
                    "architecture": architecture,
                    "protocol": protocol,
                    "metric": "accuracy",
                    "n": 5,
                    "mean": 0.84 - 0.02 * protocol_index - 0.005 * architecture_index,
                    "sd": 0.01,
                    "complete": True,
                }
            )
    svg = module.figure_svg(pd.DataFrame(rows))
    assert "mean ± sample SD over five fixed seeds" in svg
    assert "P1′ is training-size matched to P2" in svg
    assert "ViT-B/16" in svg
