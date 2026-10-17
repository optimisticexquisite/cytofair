#!/usr/bin/env python3
"""Generate every number and table the Study 2 manuscript quotes.

Numbers come only from structured outputs (provenance maps, Study 1 result
files, the power simulation, and the registered Study 2 analysis JSON/CSV).
Missing inputs produce ``\\pending`` placeholders, so the manuscript always
compiles and no value is ever typed by hand.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import pandas as pd

from cytofair.risk import structural_risk

PENDING = r"\pending"


def number(value, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return PENDING
    return f"{value:.{digits}f}"


def pvalue(value) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return PENDING
    if value < 0.001:
        return r"$<$0.001"
    if value < 0.01:
        return f"{value:.3f}"
    return f"{value:.2f}"


def load(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def get(record, *keys, default=None):
    for key in keys:
        if record is None:
            return default
        if isinstance(record, list):
            record = record[key] if isinstance(key, int) and key < len(record) else None
        else:
            record = record.get(key)
    return default if record is None else record


def contrast_macros(prefix: str, record) -> dict[str, str]:
    """Estimate, 95% CI, p, and support for one protocol_lmm.R output."""
    ci = get(record, "primary_ci95", default=[None, None])
    return {
        f"{prefix}Est": number(get(record, "primary_estimate")),
        f"{prefix}Lo": number(ci[0]),
        f"{prefix}Hi": number(ci[1]),
        f"{prefix}P": pvalue(get(record, "primary_p")),
        f"{prefix}Df": number(get(record, "df_satterthwaite"), 1),
        f"{prefix}KRp": pvalue(get(record, "p_kenward_roger")),
        f"{prefix}Supported": PENDING if record is None else ("supported" if record.get("supported") else "not supported"),
    }


def coefficient_macros(prefix: str, record, holm=None) -> dict[str, str]:
    ci = get(record, "ci95", default=[None, None])
    out = {
        f"{prefix}Est": number(get(record, "estimate")),
        f"{prefix}Lo": number(ci[0]),
        f"{prefix}Hi": number(ci[1]),
        f"{prefix}P": pvalue(get(record, "p")),
    }
    if holm is not None:
        out[f"{prefix}Pholm"] = pvalue(get(holm, "p_holm"))
        out[f"{prefix}Supported"] = "supported" if holm.get("supported") else "not supported"
    return out


def table(path: Path, caption: str, label: str, header: list[str], rows: list[list[str]], align: str) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
        *(" & ".join(row) + r" \\" for row in rows),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", default="results/study2/analysis")
    parser.add_argument("--output", default="paper/v2/generated")
    parser.add_argument("--registration-commit", default="a62e8e7")
    args = parser.parse_args()
    analysis, output = Path(args.analysis), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    macros: dict[str, str] = {}

    stamp = subprocess.check_output(
        ["git", "show", "-s", "--format=%ad", "--date=format:%Y-%m-%d %H:%M IST", args.registration_commit], text=True
    ).strip()
    macros["RegistrationCommit"] = args.registration_commit
    macros["RegistrationTime"] = stamp

    pilot = load(Path("results/study2/power/power_pilot.json"))
    macros["PilotEst"] = number(get(pilot, "pilot", "pilot_effect"))
    macros["PilotSE"] = number(get(pilot, "pilot", "pilot_effect_se"))
    macros["PilotP"] = pvalue(get(pilot, "pilot", "pilot_effect_p"))
    macros["PilotDf"] = number(get(pilot, "pilot", "pilot_effect_df"), 1)
    power = pd.read_csv("results/study2/power/power_power.csv")
    chosen = power.loc[(power.delta == 0.8) & (power.seeds == 20) & (power.architectures == 12)]
    macros["PowerChosen"] = number(float(chosen.loc[chosen.scale == 1.0, "power"].iloc[0]), 3)
    macros["PowerInflated"] = number(float(chosen.loc[chosen.scale == 1.5, "power"].iloc[0]), 3)

    h1 = load(analysis / "h1_accuracy.json")
    macros.update(contrast_macros("Hone", h1))
    boot = get(h1, "cluster_bootstrap", default={})
    macros["HoneBootLo"] = number(get(boot, "ci95", 0))
    macros["HoneBootHi"] = number(get(boot, "ci95", 1))
    macros["HoneBootP"] = pvalue(get(boot, "p_two_sided"))
    macros["HoneSignFlipP"] = pvalue(get(h1, "sign_flip", "p_two_sided"))
    macros["HoneGlmmLogOR"] = number(get(h1, "binomial_glmm", "log_odds_ratio"), 3)
    macros["HoneGlmmP"] = pvalue(get(h1, "binomial_glmm", "p_wald"))
    macros["HoneModel"] = PENDING if h1 is None else str(h1.get("model_used")).replace("_", " ")
    per_architecture = get(h1, "per_architecture_descriptive", default=[])
    macros["HonePositiveArchs"] = PENDING if not per_architecture else str(
        sum(row["mean_difference"] > 0 for row in per_architecture)
    )
    if per_architecture:
        differences = [row["mean_difference"] for row in per_architecture]
        macros["HoneArchMin"], macros["HoneArchMax"] = number(min(differences)), number(max(differences))
    else:
        macros["HoneArchMin"] = macros["HoneArchMax"] = PENDING
    macros.update(contrast_macros("PtwoPthree", load(analysis / "p2_p3_accuracy.json")))
    macros.update(contrast_macros("HoneFone", load(analysis / "h1_macro_f1.json")))
    macros.update(contrast_macros("HoneBacc", load(analysis / "h1_balanced_accuracy.json")))

    cric = load(analysis / "h2_cric_accuracy.json")
    mechanism = load(analysis / "mechanism.json")
    holm = {item["hypothesis"]: item for item in get(mechanism, "holm", default=[])}
    macros.update(contrast_macros("Htwo", cric))
    macros["HtwoPholm"] = pvalue(get(holm.get("H2"), "p_holm"))
    macros["HtwoSupported"] = PENDING if "H2" not in holm else ("supported" if holm["H2"]["supported"] else "not supported")
    macros.update(contrast_macros("CricPrime", load(analysis / "cric_p1prime_accuracy.json")))
    macros.update(contrast_macros("CricBacc", load(analysis / "cric_balanced_accuracy.json")))
    macros.update(contrast_macros("CricFone", load(analysis / "cric_macro_f1.json")))

    macros.update(coefficient_macros("Hthree", get(mechanism, "H3"), holm.get("H3")))
    macros.update(coefficient_macros("Div", get(mechanism, "diversity_effect_clean_secondary")))
    macros.update(coefficient_macros("Interaction", get(mechanism, "H3_interaction_exploratory")))
    macros.update(coefficient_macros("Hfour", get(mechanism, "H4"), holm.get("H4")))
    macros.update(coefficient_macros("LeakOriginal", get(mechanism, "H4_leak_effect_on_original_images")))
    macros.update(coefficient_macros("DonorShift", get(mechanism, "background_class_information_secondary")))
    macros.update(coefficient_macros("HfourBg", get(mechanism, "H4_background_ge_0.30_secondary")))
    h4, leak_original = get(mechanism, "H4", "estimate"), get(mechanism, "H4_leak_effect_on_original_images", "estimate")
    macros["RouteShare"] = (
        PENDING if h4 is None or leak_original in (None, 0) else number(100 * h4 / leak_original, 0)
    )
    macros.update(coefficient_macros("Hfive", get(mechanism, "H5"), holm.get("H5")))
    macros["HfiveStatus"] = PENDING if mechanism is None else str(get(mechanism, "H5", "status", default="tested"))

    secondary = load(analysis / "secondary.json")
    macros["MoneEst"] = number(get(secondary, "M1", "mean_within_crop_difference_pp"))
    macros["MoneLo"] = number(get(secondary, "M1", "ci95_pp", 0))
    macros["MoneHi"] = number(get(secondary, "M1", "ci95_pp", 1))
    macros["MoneCrops"] = PENDING if secondary is None else f"{get(secondary, 'M1', 'crops'):,}"
    macros["MtwoOverlapLogOdds"] = number(get(secondary, "M2", "overlap", "log_odds"))
    macros["MtwoOverlapP"] = pvalue(get(secondary, "M2", "overlap", "p"))
    macros["MtwoSiblingsLogOdds"] = number(get(secondary, "M2", "log_siblings", "log_odds"))
    macros["MtwoSiblingsP"] = pvalue(get(secondary, "M2", "log_siblings", "p"))
    ranking = get(secondary, "ranking_P1_vs_P2", 0)
    macros["KendallTau"] = number(get(ranking, "effect_size"))
    macros["KendallLo"] = number(get(ranking, "effect_ci_low"))
    macros["KendallHi"] = number(get(ranking, "effect_ci_high"))
    macros["ReorderedPairs"] = PENDING if ranking is None else f"{ranking['reordered_pairs']}/{ranking['total_pairs']}"
    ordering = get(secondary, "ordering_P3_le_P2_le_P1", default=[])
    macros["OrderingPassed"] = PENDING if not ordering else f"{sum(bool(r['ordering_passed']) for r in ordering)}/{len(ordering)}"
    decomposition = get(secondary, "approximate_gap_decomposition", default={})
    macros["DecompGap"] = number(get(decomposition, "phase1_gap_pp"))
    macros["DecompLeak"] = number(get(decomposition, "leakage_part_pp"))
    macros["DecompDiv"] = number(get(decomposition, "diversity_part_pp"))

    lines = [r"\providecommand{\pending}{\textit{[pending]}}"]
    for name, value in sorted(macros.items()):
        lines.append(rf"\newcommand{{\{name}}}{{{value}}}")
    (output / "numbers.tex").write_text("\n".join(lines) + "\n")

    maps = {
        "SIPaKMeD": ("provenance/sipakmed_parent_map_p3.parquet", "cell crop", "cluster field", "5-class classification"),
        "Cx22": ("provenance/cx22_parent_map.parquet", "generated ROI", "CCEDD source image", "segmentation"),
        "CRIC": ("provenance/cric_parent_map.parquet", "classified cell", "slide image", "6-class classification"),
        "Herlev": ("provenance/herlev_parent_map.parquet", "single-cell image", "the image itself", "7-class classification"),
    }
    risks, rows = {}, []
    for name, (path, unit, parent, task) in maps.items():
        frame = pd.read_parquet(path)
        risks[name] = structural_risk(frame)
        rows.append([name, unit, parent, f"{len(frame):,}", f"{frame.parent_id.nunique():,}",
                     f"{risks[name]['crops_per_parent_mean']:.2f}", task])
    table(output / "table_datasets.tex", "Audited datasets and their independent units.", "tab:datasets",
          ["Dataset", "Observation", "Parent", "Obs.", "Parents", "Obs./parent", "Released task"], rows, "lllrrrl")

    structure = pd.read_csv("results/table1_structural_leakage.csv").set_index("dataset")
    overlap = pd.read_csv("results/study2/structural/literal_overlap_p1.csv")
    overlap = overlap.groupby("dataset").test_crops_with_literal_overlap.agg(["mean", "std"]).mul(100)
    rows = []
    for name in maps:
        risk = risks[name]
        dependence = risk["class_parent_dependence"]
        literal = (
            rf"{overlap.loc[name, 'mean']:.2f}\,$\pm$\,{overlap.loc[name, 'std']:.2f}"
            if name in overlap.index
            else "0.00"
        )
        rows.append([
            name,
            f"{structure.loc[name, 'analytic_p1_straddle_pct']:.1f}",
            structure.loc[name, "observed_p1_straddle_pct_mean_sd"].replace(" ± ", r"\,$\pm$\,"),
            structure.loc[name, "observed_p1_test_leak_pct_mean_sd"].replace(" ± ", r"\,$\pm$\,"),
            f"{risk['expected_training_siblings_per_test_crop']:.1f}",
            f"{risk['training_diversity_ratio']:.2f}",
            "--" if not math.isfinite(dependence) or risk["crops_per_parent_mean"] == 1 else f"{dependence:.2f}",
            literal,
        ])
    table(
        output / "table_structure.tex",
        "Structural leakage of a random 80:20 crop split. Straddle and test-crop leakage are percentages "
        "(observed: mean\\,$\\pm$\\,SD over five seeds); siblings, diversity ratio, and class--parent dependence "
        "are closed-form expectations (\\texttt{cytofair risk-report}); literal overlap is the percentage of "
        "test observations sharing source pixels with a training observation (same five seeds). P2 and P3 are exactly zero on every column.",
        "tab:structure",
        ["Dataset", "Analytic", "Observed", "Test w/ train parent", "Siblings", "Diversity", "Class dep.", "Literal"],
        rows,
        "lrrrrrrr",
    )
    print(f"wrote {len(macros)} macros and tables to {output}")


if __name__ == "__main__":
    main()
