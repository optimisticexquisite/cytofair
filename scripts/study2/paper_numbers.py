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
    text = f"{abs(value):.{digits}f}"
    # A typographic minus, and no sign on values that round to zero.
    return r"\textminus{}" + text if value < 0 and float(text) != 0 else text


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


def descriptive_macros(analysis: Path, mechanism, h1) -> dict[str, str]:
    """Plain descriptive numbers the text quotes (accuracy levels, checks, condition contrasts)."""
    out: dict[str, str] = {}
    keys = ("SipPOneAcc", "SipPTwoAcc", "SipErrorRatio", "CricPOneAcc", "CricPTwoAcc", "CricPOneBacc", "CricPTwoBacc",
            "CricErrorRatio", "HoneArchCIPositive", "GrlConvRzero", "GrlConvRstar", "GrlResRzero", "GrlResRstar",
            "GrlResGapZero", "GrlResGapMin", "GrlResGapMax", "TransplantLeakUnseen", "TransplantLeakOther",
            "TransplantCleanHarmony", "TransplantLeakSibling")
    out.update(dict.fromkeys(keys, PENDING))
    phase1_path, cric_path = analysis / "runs" / "phase1_runs.csv", analysis / "runs" / "cric_runs.csv"
    if phase1_path.is_file():
        means = pd.read_csv(phase1_path).groupby("protocol").accuracy.mean().mul(100)
        out["SipPOneAcc"], out["SipPTwoAcc"] = number(means["P1"]), number(means["P2"])
        out["SipErrorRatio"] = number((100 - means["P2"]) / (100 - means["P1"]))
    if cric_path.is_file():
        cric = pd.read_csv(cric_path).groupby("protocol")[["accuracy", "balanced_accuracy"]].mean().mul(100)
        out["CricPOneAcc"], out["CricPTwoAcc"] = number(cric.accuracy["P1"]), number(cric.accuracy["P2"])
        out["CricPOneBacc"], out["CricPTwoBacc"] = number(cric.balanced_accuracy["P1"]), number(cric.balanced_accuracy["P2"])
        out["CricErrorRatio"] = number((100 - cric.accuracy["P2"]) / (100 - cric.accuracy["P1"]))
    per_architecture = get(h1, "per_architecture_descriptive", default=[])
    if per_architecture:
        out["HoneArchCIPositive"] = str(sum(row["ci_low"] > 0 for row in per_architecture))
    checks = get(mechanism, "H5_manipulation_checks", default={})
    for arch, prefix in (("convnext_tiny", "GrlConv"), ("resnet50", "GrlRes")):
        if arch in checks:
            out[f"{prefix}Rzero"] = number(checks[arch]["R_baseline"], 1)
            out[f"{prefix}Rstar"] = number(checks[arch]["R_star"], 1)
    grl_path = analysis / "mechanism" / "grl_recoverability.csv"
    if grl_path.is_file():
        grl = pd.read_csv(grl_path)
        res = grl.loc[grl.architecture == "resnet50"]
        wide = res.pivot_table(index=["seed", "adversary_lambda"], columns="protocol", values="accuracy")
        gaps = (100 * (wide.P1 - wide.P2)).groupby("adversary_lambda").mean()
        out["GrlResGapZero"] = number(gaps.loc[0.0])
        out["GrlResGapMin"], out["GrlResGapMax"] = number(gaps.drop(0.0).min()), number(gaps.drop(0.0).max())
    transplant_path = analysis / "mechanism" / "transplant_runs.csv"
    if transplant_path.is_file():
        table = pd.read_csv(transplant_path)
        table = table.loc[table.subset == "all"]
        leak = table.cell.str.startswith("leak")
        difference = lambda column: 100 * (table.loc[leak, column].mean() - table.loc[~leak, column].mean())
        out["TransplantLeakSibling"] = number(difference("acc_sibling"))
        out["TransplantLeakUnseen"] = number(difference("acc_unseen_same_class"))
        out["TransplantLeakOther"] = number(difference("acc_unseen_other_class"))
        clean = table.loc[~leak]
        out["TransplantCleanHarmony"] = number(100 * (clean.acc_sibling - clean.acc_unseen_same_class).mean())
    return out


def followup_macros(root: Path) -> dict[str, str]:
    """Amendment 8 numbers, or placeholders while the follow-up matrix runs."""
    out: dict[str, str] = {}
    cric = load(root / "cric_context.json")
    out["FoneEst"] = number(get(cric, "F1", "estimate"))
    out["FoneLo"], out["FoneHi"] = number(get(cric, "F1", "ci95", 0)), number(get(cric, "F1", "ci95", 1))
    out["FoneP"] = pvalue(get(cric, "F1", "p_two_sided"))
    out["FoneCells"] = PENDING if cric is None else f"{cric['F1']['cells']:,}"
    out["FoneSlides"] = PENDING if cric is None else str(cric["F1"]["slides"])
    out["FoneConcordant"] = number(get(cric, "accuracy_difference_concordant", "estimate"))
    out["FoneDiscordant"] = number(get(cric, "accuracy_difference_discordant", "estimate"))
    out["FoneContrast"] = number(get(cric, "concordance_contrast_secondary", "estimate"))
    out["FoneContrastLo"] = number(get(cric, "concordance_contrast_secondary", "ci95", 0))
    out["FoneContrastHi"] = number(get(cric, "concordance_contrast_secondary", "ci95", 1))
    followup = load(root / "followup.json")
    holm = {item["hypothesis"]: item for item in get(followup, "holm", default=[])}
    letters = {"F1": "Fone", "F2": "Ftwo", "F3": "Fthree", "F4": "Ffour", "F5": "Ffive", "F6": "Fsix",
               "F7": "Fseven", "F8": "Feight"}
    for key, prefix in letters.items():
        item = holm.get(key)
        out[f"{prefix}Pholm"] = pvalue(get(item, "p_holm"))
        out[f"{prefix}Supported"] = PENDING if item is None else ("supported" if item["supported"] else "not supported")
        if key == "F1":
            continue
        record = get(followup, key)
        out.update({k: v for k, v in coefficient_macros(prefix, record).items()})
        out[f"{prefix}Status"] = PENDING if record is None else str(record.get("status", "tested"))
    for name, prefix in (("grey", "AblGrey"), ("low_pass", "AblLow"), ("illumination_flat", "AblFlat"),
                         ("background_grey", "AblBg")):
        out.update(coefficient_macros(prefix, get(followup, "ablation_reductions_secondary", name)))
    out.update(coefficient_macros("HthreeRep", get(followup, "H3_replication_secondary")))
    out.update(coefficient_macros("MaskReduction", get(followup, "masking_reduction_secondary")))
    out.update(coefficient_macros("CamLeak", get(followup, "gradcam_descriptive", "leak_effect_on_background_attribution")))
    return out


def table(path: Path, caption: str, label: str, header: list[str], rows: list[list[str]], align: str) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
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
    parser.add_argument("--followup", default="results/study2/followup")
    parser.add_argument("--followup-commit", default="80171fe")
    args = parser.parse_args()
    analysis, output = Path(args.analysis), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    macros: dict[str, str] = {}

    stamp = subprocess.check_output(
        ["git", "show", "-s", "--format=%ad", "--date=format:%Y-%m-%d %H:%M IST", args.registration_commit], text=True
    ).strip()
    macros["RegistrationCommit"] = args.registration_commit
    macros["RegistrationTime"] = stamp
    macros["FollowupCommit"] = args.followup_commit
    macros["FollowupTime"] = subprocess.check_output(
        ["git", "show", "-s", "--format=%ad", "--date=format:%Y-%m-%d %H:%M IST", args.followup_commit], text=True
    ).strip()

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

    macros.update(descriptive_macros(analysis, mechanism, h1))
    macros.update(followup_macros(Path(args.followup)))

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
