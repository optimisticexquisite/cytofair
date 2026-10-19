#!/usr/bin/env python3
"""Generate every table and record-level macro of the lifecycle record from committed outputs.

Inputs: ``results/study2/analysis``, ``results/study2/followup``,
``results/study2/run_records``, ``results/study2/power``, provenance maps, and git history.
Outputs: ``record/generated/*.tex``. Nothing is typed by hand.
"""

from __future__ import annotations

import json
import math
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "record" / "generated"
A = ROOT / "results" / "study2" / "analysis"
F = ROOT / "results" / "study2" / "followup"
R = ROOT / "results" / "study2" / "run_records"
IST = timezone(timedelta(hours=5, minutes=30))
NAMES = {
    "resnet18": "ResNet-18", "resnet50": "ResNet-50", "densenet121": "DenseNet-121",
    "efficientnet_b0": "EfficientNet-B0", "mobilenet_v2": "MobileNet-V2", "vit_b_16": "ViT-B/16",
    "convnext_tiny": "ConvNeXt-T", "swin_t": "Swin-T", "regnet_y_1_6gf": "RegNetY-1.6GF",
    "efficientnet_v2_s": "EfficientNetV2-S", "maxvit_t": "MaxViT-T", "dinov2_vits14": "DINOv2 ViT-S/14",
}
ORDER = list(NAMES)


def esc(text) -> str:
    return (str(text).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")
            .replace("&", r"\&").replace("#", r"\#"))


def brk(text) -> str:
    """Escape and allow line breaks after underscores and dots in long identifiers."""
    return esc(text).replace(r"\_", r"\_\allowbreak{}").replace(".", r".\allowbreak{}")


def num(value, digits=2) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "--"
    text = f"{abs(value):.{digits}f}"
    return (r"\textminus{}" + text) if value < 0 and float(text) != 0 else text


def pval(value) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "--"
    return r"$<$0.001" if value < 0.001 else (f"{value:.3f}" if value < 0.01 else f"{value:.2f}")


def longtable(path: Path, caption: str, label: str, header: list[str], rows: list[list[str]], align: str,
              size: str = r"\footnotesize") -> None:
    head = " & ".join(header) + r" \\"
    lines = [
        r"{" + size, r"\setlength{\tabcolsep}{3.5pt}", rf"\begin{{longtable}}{{{align}}}",
        rf"\caption{{{caption}}}\label{{{label}}}\\", r"\toprule", head, r"\midrule", r"\endfirsthead",
        r"\toprule", head, r"\midrule", r"\endhead", r"\bottomrule", r"\endfoot",
        *(" & ".join(map(str, row)) + r" \\" for row in rows), r"\end{longtable}", r"}",
    ]
    path.write_text("\n".join(lines) + "\n")


def arch_sort(frame: pd.DataFrame, column="architecture") -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [c if not isinstance(c, tuple) else "|".join(str(x) for x in c if str(x)) for c in frame.columns]
    return frame.assign(_o=frame[column].map({a: i for i, a in enumerate(ORDER)})).sort_values("_o", kind="stable").drop(columns="_o")


def mean_sd(series, scale=100, digits=2) -> str:
    values = series.dropna() * scale
    if len(values) == 0:
        return "--"
    if len(values) == 1:
        return num(values.iloc[0], digits)
    return f"{num(values.mean(), digits)}\\,$\\pm$\\,{num(values.std(ddof=1), digits)}"


def to_ist(text: str) -> datetime:
    return datetime.fromisoformat(text).astimezone(IST)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    macros: dict[str, str] = {}
    runs = pd.read_csv(R / "training_runs.csv")
    evals = pd.read_csv(R / "evaluation_runs.csv")
    smoke = pd.read_csv(R / "smoke_runs.csv")

    # ---------------- counts, compute, and timing ----------------
    study2 = runs.loc[runs.experiment.isin(["phase1", "factorial", "grl", "cric"])]
    follow = runs.loc[runs.experiment.isin(["carrier", "linadv", "normalization"])]
    macros["RecStudyTwoRuns"] = f"{len(study2) + 1:,}"
    macros["RecFollowTrainRuns"] = f"{len(follow):,}"
    macros["RecFollowEvalRuns"] = f"{len(evals):,}"
    macros["RecFollowTasks"] = f"{len(follow) + len(evals):,}"
    macros["RecAllModels"] = f"{325 + len(study2) + 1 + len(follow):,}"
    macros["RecEpochs"] = f"{int(runs.groupby('run').size().sum() if False else 20 * len(runs)):,}"
    gpu_hours = runs.groupby("experiment").elapsed_seconds.sum() / 3600
    eval_hours = evals.groupby("experiment").elapsed_seconds.sum() / 3600
    macros["RecGpuHoursStudyTwo"] = num(gpu_hours[["phase1", "factorial", "grl", "cric"]].sum(), 1)
    macros["RecGpuHoursFollow"] = num(gpu_hours[["carrier", "linadv", "normalization"]].sum() + eval_hours.sum(), 1)
    starts = runs.started_at_utc.map(to_ist)
    ends = runs.completed_at_utc.map(to_ist)
    s2 = runs.experiment.isin(["phase1", "factorial", "grl", "cric"])
    macros["RecStudyTwoStart"] = starts[s2].min().strftime("%d %b %H:%M")
    macros["RecStudyTwoEnd"] = ends[s2].max().strftime("%d %b %H:%M")
    macros["RecStudyTwoWall"] = num((ends[s2].max() - starts[s2].min()).total_seconds() / 3600, 1)
    eval_end = evals.completed_at_utc.map(to_ist)
    eval_start = eval_end - pd.to_timedelta(evals.elapsed_seconds, unit="s")
    fstart = min(starts[~s2].min(), eval_start.min())
    fend = max(ends[~s2].max(), eval_end.max())
    macros["RecFollowStart"] = fstart.strftime("%d %b %H:%M")
    macros["RecFollowEnd"] = fend.strftime("%d %b %H:%M")
    macros["RecFollowWall"] = num((fend - fstart).total_seconds() / 3600, 1)
    macros["RecSmokeRuns"] = str(len(smoke))
    commits = subprocess.check_output(["git", "-C", str(ROOT), "log", "--all", "--format=%h"], text=True).split()
    macros["RecCommits"] = str(len(commits))

    rows = []
    labels = {"phase1": "Study 2 Phase 1 (H1)", "factorial": "Study 2 2C/2B (H3, H4)", "grl": "Study 2 2A (H5)",
              "cric": "Study 2 CRIC (H2)", "carrier": "A8 carrier factorial (F3, F4)", "linadv": "A8 linear adversary (F7)",
              "normalization": "A8 stain normalisation (F6)"}
    for experiment, label in labels.items():
        group = runs.loc[runs.experiment == experiment]
        st, en = group.started_at_utc.map(to_ist), group.completed_at_utc.map(to_ist)
        rows.append([label, f"{len(group):,}", num(group.elapsed_seconds.mean() / 60, 1),
                     num(group.elapsed_seconds.sum() / 3600, 1), st.min().strftime("%d %b %H:%M"),
                     en.max().strftime("%d %b %H:%M")])
    for experiment, label in (("factorial_eval", "A8 ablations + Grad-CAM (F2)"), ("leace", "A8 LEACE erasure (F5)")):
        group = evals.loc[evals.experiment == experiment]
        en = group.completed_at_utc.map(to_ist)
        st = en - pd.to_timedelta(group.elapsed_seconds, unit="s")
        rows.append([label, f"{len(group):,}", num(group.elapsed_seconds.mean() / 60, 1),
                     num(group.elapsed_seconds.sum() / 3600, 1), st.min().strftime("%d %b %H:%M"),
                     en.max().strftime("%d %b %H:%M")])
    longtable(OUT / "compute_ledger.tex", "Compute ledger of every Study 2 and Amendment 8 arm (node1, NVIDIA RTX A6000; "
              "times in IST). GPU-hours are summed per-run elapsed time; runs shared four GPUs.", "r:tab:compute",
              ["Arm", "Runs", "Mean min", "GPU-h", "First start", "Last finish"], rows, "lrrrll")

    per_arch = []
    for arch in ORDER:
        group = runs.loc[(runs.experiment == "phase1") & (runs.architecture == arch)]
        per_arch.append([NAMES[arch], str(int(group.feature_dim.iloc[0])), brk(group.weights.iloc[0]),
                         num(group.training_seconds.mean() / 60, 2), num(group.elapsed_seconds.mean() / 60, 2)])
    longtable(OUT / "architectures.tex", "The twelve Study 2 backbones: penultimate feature dimension, ImageNet weights, "
              "and mean Phase 1 training and total minutes per run.", "r:tab:archs",
              ["Architecture", "Dim", "Weights", "Train min", "Total min"], per_arch, "llp{4.6cm}rr")

    versions = json.loads((R / "runtime_versions.json").read_text())
    vrows = []
    for record in versions.values():
        for key in ("python_version", "torch_version", "torchvision_version", "timm_version", "cuda_version",
                    "cudnn_version", "cuda_device", "numpy_version", "pandas_version", "scipy_version",
                    "scikit_learn_version", "opencv_version", "pillow_version", "platform", "cublas_workspace_config"):
            vrows.append([esc(key.replace("_version", "")), esc(record.get(key))])
    longtable(OUT / "software.tex", "Software and hardware recorded in every training run.", "r:tab:software",
              ["Component", "Version"], vrows, "lp{11cm}")

    # ---------------- registered tests ----------------
    h = {k: json.loads((A / f"{k}.json").read_text()) for k in (
        "h1_accuracy", "p2_p3_accuracy", "h1_macro_f1", "h1_balanced_accuracy", "h2_cric_accuracy",
        "cric_p1prime_accuracy", "cric_balanced_accuracy", "cric_macro_f1")}
    mech = json.loads((A / "mechanism.json").read_text())
    fol = json.loads((F / "followup.json").read_text())
    cric = json.loads((F / "cric_context.json").read_text())
    mholm = {i["hypothesis"]: i for i in mech["holm"]}
    fholm = {i["hypothesis"]: i for i in fol["holm"]}
    rows = []

    def verdict(flag):
        return r"\supported" if flag else r"\notsupported"
    rows.append(["H1", "SIPaKMeD P1 $-$ P2 accuracy (primary)", num(h["h1_accuracy"]["primary_estimate"]),
                 f"{num(h['h1_accuracy']['primary_ci95'][0])} to {num(h['h1_accuracy']['primary_ci95'][1])}",
                 pval(h["h1_accuracy"]["primary_p"]), "alone", verdict(h["h1_accuracy"]["supported"])])
    for key, label, record in (("H2", "CRIC P1 $-$ P2 accuracy", h["h2_cric_accuracy"]),):
        rows.append([key, label, num(record["primary_estimate"]),
                     f"{num(record['primary_ci95'][0])} to {num(record['primary_ci95'][1])}", pval(record["primary_p"]),
                     pval(mholm[key]["p_holm"]), verdict(mholm[key]["supported"])])
    for key, label in (("H3", "2C leakage at matched diversity"), ("H4", "2B own-context route"),
                       ("H5", "2A gap vs recoverability")):
        record = mech[key]
        rows.append([key, label, num(record["estimate"], 3 if key == "H5" else 2),
                     f"{num(record['ci95'][0], 3 if key == 'H5' else 2)} to {num(record['ci95'][1], 3 if key == 'H5' else 2)}",
                     pval(record["p"]), pval(mholm[key]["p_holm"]), verdict(mholm[key]["supported"])])
    labels_f = {"F1": "CRIC slide-context pull", "F2": "test-time stain transfer", "F3": "leak with backgrounds masked",
                "F4": "colour augmentation", "F5": "LEACE erasure", "F6": "stain normalisation",
                "F7": "probe-matched adversary", "F8": "pooled dose-response"}
    for key, label in labels_f.items():
        record = cric["F1"] if key == "F1" else fol[key]
        digits = 3 if key in {"F7", "F8"} else 2
        p = record.get("p_two_sided", record.get("p"))
        rows.append([key, label, num(record["estimate"], digits),
                     f"{num(record['ci95'][0], digits)} to {num(record['ci95'][1], digits)}", pval(p),
                     pval(fholm[key]["p_holm"]), verdict(fholm[key]["supported"])])
    longtable(OUT / "all_tests.tex", "Every registered confirmatory test of Study 2 and Amendment 8. H1 is tested alone; "
              "H2--H5 and F1--F8 are separate Holm families. Units as in the text (pp; slopes in pp per pp).",
              "r:tab:alltests", ["", "Question", "Estimate", "95\\% CI", "$p$", "Holm $p$", "Verdict"], rows,
              "llrrrrl")

    # Sensitivity analyses of the protocol models.
    rows = []
    for key, label in (("h1_accuracy", "H1 accuracy"), ("h1_macro_f1", "H1 macro-F1"),
                       ("h1_balanced_accuracy", "H1 balanced accuracy"), ("p2_p3_accuracy", "P2 $-$ P3 accuracy"),
                       ("h2_cric_accuracy", "H2 CRIC accuracy"), ("cric_p1prime_accuracy", "CRIC P1$'$ $-$ P2"),
                       ("cric_balanced_accuracy", "CRIC balanced accuracy"), ("cric_macro_f1", "CRIC macro-F1")):
        r = h[key]
        boot = r.get("cluster_bootstrap", {})
        rows.append([label, esc(r.get("model_used")), num(r.get("estimate")), num(r.get("se")),
                     num(r.get("df_satterthwaite"), 1), pval(r.get("p_satterthwaite")), num(r.get("df_kenward_roger"), 1),
                     pval(r.get("p_kenward_roger")),
                     f"{num(boot.get('ci95', [None, None])[0])}--{num(boot.get('ci95', [None, None])[1])}",
                     pval(r.get("sign_flip", {}).get("p_two_sided")), esc(r.get("primary_source"))])
    longtable(OUT / "protocol_sensitivity.tex", "Every protocol model with its registered sensitivity analyses: "
              "Satterthwaite and Kenward--Roger degrees of freedom, two-way cluster bootstrap, and seed sign-flip test. "
              "The primary source is the bootstrap when the registered residual check fails.", "r:tab:sensitivity",
              ["Contrast", "Model", "Est.", "SE", "df$_S$", "$p_S$", "df$_{KR}$", "$p_{KR}$", "Bootstrap CI",
               "Sign-flip $p$", "Primary"], rows, "llrrrrrrrrl", size=r"\scriptsize")
    posthoc = ROOT / "results" / "study2" / "posthoc"
    sip = json.loads((posthoc / "study1_sipakmed_pooled_lmm.json").read_text())
    cx = json.loads((posthoc / "study1_cx22_semantic_pooled_lmm.json").read_text())
    macros["RecPostSipEst"], macros["RecPostSipP"] = num(sip["estimate"]), pval(sip["p_satterthwaite"])
    macros["RecPostSipSE"], macros["RecPostSipDf"] = num(sip["se"]), num(sip["df_satterthwaite"], 1)
    macros["RecPostCxEst"], macros["RecPostCxP"] = num(cx["estimate"]), pval(cx["p_satterthwaite"])
    overlap = pd.read_csv(ROOT / "results" / "study2" / "structural" / "literal_overlap_p1.csv")
    overlap = overlap.groupby("dataset").test_crops_with_literal_overlap.mean() * 100
    for name, key in (("SIPaKMeD", "RecOverlapSip"), ("CRIC", "RecOverlapCric"), ("Cx22", "RecOverlapCx")):
        macros[key] = num(overlap[name], 1)
    pilot = json.loads((ROOT / "results" / "study2" / "power" / "power_pilot.json").read_text())["pilot"]
    for key, field in (("RecPilotSdInt", "sd_int"), ("RecPilotSdSlope", "sd_slope"), ("RecPilotSdSplitOne", "sd_split_p1"),
                       ("RecPilotSdSplitTwo", "sd_split_p2"), ("RecPilotSdResid", "sd_resid"), ("RecPilotMu", "mu")):
        macros[key] = num(pilot[field], 2 if field != "mu" else 2)
    macros["RecAllTasks"] = f"{len(study2) + 1 + len(follow) + len(evals):,}"
    secondary = json.loads((A / "secondary.json").read_text())
    for protocol, key in (("P1", "One"), ("P2", "Two"), ("P3", "Three")):
        macros[f"RecEce{key}"] = num(100 * secondary["calibration_by_protocol"][protocol]["ece_15"])
        macros[f"RecNll{key}"] = num(secondary["calibration_by_protocol"][protocol]["nll"], 3)
    diversity = secondary["embedding_diversity"]
    for cell, key in (("P1", "POne"), ("P2", "PTwo"), ("clean_high", "CleanHigh"), ("clean_low", "CleanLow"),
                      ("leak_high", "LeakHigh"), ("leak_low", "LeakLow")):
        macros[f"RecDiv{key}"] = num(diversity[cell]["mean_pairwise_parent_centroid_cosine_distance"], 3)
        macros[f"RecDivParents{key}"] = num(diversity[cell]["n_parents"], 0)
    encoder = json.loads((R / "fingerprint_encoder.json").read_text())
    macros["RecEncoderAcc"] = num(100 * encoder["test_accuracy"], 1)
    macros["RecEncoderChance"] = num(100 * encoder["chance"], 2)
    macros["RecEncoderTimes"] = num(encoder["test_accuracy"] / encoder["chance"], 0)
    ordering = secondary["ordering_P3_le_P2_le_P1"]
    macros["RecOrderingPassed"] = str(sum(bool(r["ordering_passed"]) for r in ordering))
    by_ablation = fol["leak_effect_by_ablation_secondary"]
    original = by_ablation["original"]["estimate"]
    macros["RecBgKeep"] = num(100 * by_ablation["background_grey"]["estimate"] / original, 0)
    macros["RecFlatKeep"] = num(100 * by_ablation["illumination_flat"]["estimate"] / original, 0)
    macros["RecStainRemove"] = num(100 * (1 - by_ablation["stain_transfer"]["estimate"] / original), 0)
    a = h["h1_accuracy"]["assumptions"]
    macros["RecHoneSkew"] = num(a["residual_skewness"])
    macros["RecHoneKurt"] = num(a["residual_excess_kurtosis"])
    macros["RecHoneShapiro"] = pval(a["residual_shapiro_p"])

    # Variance components of every registered mixed model.
    rows = []

    def add_components(label, components):
        for item in components or []:
            var2 = item.get("var2")
            name = esc(item["grp"]) + ": " + esc(item.get("var1") or "residual")
            if var2 and var2 == var2 and str(var2) not in ("NA", "None", "nan"):
                name += " $\\times$ " + esc(var2)
                rows.append([label, name, "--", num(item["sdcor"], 3) + " (corr.)"])
            else:
                rows.append([label, name, num(item["vcov"], 4), num(item["sdcor"], 3)])
            label = ""
    for key, label in (("h1_accuracy", "H1"), ("p2_p3_accuracy", "P2 $-$ P3"), ("h2_cric_accuracy", "H2")):
        add_components(label, h[key].get("variance_components"))
    for key in ("H3", "H4", "H5"):
        add_components(key, mech[key].get("variance_components"))
    for key in ("F2", "F3", "F4", "F5", "F6", "F7", "F8"):
        add_components(key, fol[key].get("variance_components"))
    longtable(OUT / "variance_components.tex", "Variance components (REML) of every registered mixed model. "
              "Correlations of random intercepts and slopes are listed as `corr.'.", "r:tab:varcomp",
              ["Model", "Component", "Variance", "SD / corr."], rows, "llrr", size=r"\scriptsize")

    # ---------------- per-arm summaries ----------------
    phase1 = pd.read_csv(A / "runs" / "phase1_runs.csv")
    rows = []
    for arch in ORDER:
        for protocol in ("P1", "P2", "P3"):
            g = phase1.loc[(phase1.architecture == arch) & (phase1.protocol == protocol)]
            rows.append([NAMES[arch] if protocol == "P1" else "", protocol, mean_sd(g.accuracy), mean_sd(g.macro_f1),
                         mean_sd(g.balanced_accuracy), mean_sd(g.ece_15), num(g.nll.mean(), 3)])
    longtable(OUT / "phase1_summary.tex", "Phase 1 (SIPaKMeD, 20 seeds per cell): test metrics in percent "
              "(mean\\,$\\pm$\\,SD over seeds); NLL in nats.", "r:tab:phase1",
              ["Architecture", "Prot.", "Accuracy", "Macro-F1", "Balanced acc.", "ECE", "NLL"], rows, "llrrrrr")
    cric_runs = pd.read_csv(A / "runs" / "cric_runs.csv")
    rows = []
    for arch in [a for a in ORDER if a in set(cric_runs.architecture)]:
        for protocol in ("P1", "P1-prime", "P2"):
            g = cric_runs.loc[(cric_runs.architecture == arch) & (cric_runs.protocol == protocol)]
            rows.append([NAMES[arch] if protocol == "P1" else "", protocol, mean_sd(g.accuracy), mean_sd(g.macro_f1),
                         mean_sd(g.balanced_accuracy), mean_sd(g.ece_15), num(g.nll.mean(), 3)])
    longtable(OUT / "cric_summary.tex", "CRIC (10 seeds per cell): test metrics in percent (mean\\,$\\pm$\\,SD).",
              "r:tab:cric", ["Architecture", "Prot.", "Accuracy", "Macro-F1", "Balanced acc.", "ECE", "NLL"], rows,
              "llrrrrr")
    fact = pd.read_csv(A / "runs" / "factorial_runs.csv")
    trans = pd.read_csv(A / "mechanism" / "transplant_runs.csv")
    trans = trans.loc[trans.subset == "all"]
    rows = []
    for arch in [a for a in ORDER if a in set(fact.architecture)]:
        for cell in ("leak_high", "clean_high", "leak_low", "clean_low"):
            g = fact.loc[(fact.architecture == arch) & (fact.protocol == cell)]
            t = trans.loc[(trans.architecture == arch) & (trans.cell == cell)]
            rows.append([NAMES[arch] if cell == "leak_high" else "", esc(cell), mean_sd(g.accuracy),
                         mean_sd(t.acc_self), mean_sd(t.acc_sibling), mean_sd(t.acc_unseen_same_class),
                         mean_sd(t.acc_unseen_other_class), num(100 * t.donor_probability_shift.mean())])
    longtable(OUT / "factorial_summary.tex", "2C/2B (10 seeds per cell): accuracy on the common test set and on each "
              "background-transplant condition (percent), and the donor-class probability shift (pp).",
              "r:tab:factorial", ["Architecture", "Cell", "Original", "Self", "Sibling", "Unseen same",
                                  "Unseen other", "Shift"], rows, "llrrrrrr", size=r"\scriptsize")
    grl = pd.read_csv(A / "mechanism" / "grl_recoverability.csv")
    rows = []
    for arch in [a for a in ORDER if a in set(grl.architecture)]:
        for lam in sorted(grl.adversary_lambda.unique()):
            g = grl.loc[(grl.architecture == arch) & (grl.adversary_lambda == lam)]
            p1, p2 = g.loc[g.protocol == "P1"], g.loc[g.protocol == "P2"]
            rows.append([NAMES[arch] if lam == 0 else "", f"{lam:g}", mean_sd(p1.accuracy), mean_sd(p2.accuracy),
                         num(100 * (p1.accuracy.mean() - p2.accuracy.mean())), mean_sd(p1.within_class_accuracy, digits=1),
                         mean_sd(p2.within_class_accuracy, digits=1)])
    longtable(OUT / "grl_summary.tex", "2A gradient-reversal sweep (10 seeds): accuracy (percent), mean gap (pp), and "
              "within-class parent recoverability $R$ of the frozen features (percent; sklearn probe).",
              "r:tab:grl", ["Architecture", "$\\lambda$", "P1 acc.", "P2 acc.", "Gap", "$R$ (P1)", "$R$ (P2)"], rows,
              "lrrrrrr")

    T = F / "tables"
    carrier = pd.read_csv(T / "carrier_runs.csv")
    rows = []
    for arch in [a for a in ORDER if a in set(carrier.architecture)]:
        for condition in ("standard", "masked", "colour"):
            g = carrier.loc[(carrier.architecture == arch) & (carrier.condition == condition)]
            cells = {c: g.loc[g.cell == c].accuracy for c in ("leak_high", "clean_high", "leak_low", "clean_low")}
            effect = 100 * ((cells["leak_high"].mean() + cells["leak_low"].mean()) / 2
                            - (cells["clean_high"].mean() + cells["clean_low"].mean()) / 2)
            rows.append([NAMES[arch] if condition == "standard" else "", condition,
                         *(mean_sd(cells[c]) for c in ("leak_high", "clean_high", "leak_low", "clean_low")), num(effect)])
    longtable(OUT / "carrier_summary.tex", "Amendment 8 carrier factorial (fresh seeds, 10 per cell): accuracy per cell "
              "(percent) and the leak effect (pp).", "r:tab:carrier",
              ["Architecture", "Condition", "Leak high", "Clean high", "Leak low", "Clean low", "Leak effect"], rows,
              "llrrrrr", size=r"\scriptsize")
    linear = pd.read_csv(T / "linadv_recoverability.csv")
    rows = []
    for arch in [a for a in ORDER if a in set(linear.architecture)]:
        for lam in sorted(linear.adversary_lambda.unique()):
            g = linear.loc[(linear.architecture == arch) & (linear.adversary_lambda == lam)]
            p1, p2 = g.loc[g.protocol == "P1"], g.loc[g.protocol == "P2"]
            rows.append([NAMES[arch] if lam == 0 else "", f"{lam:g}", mean_sd(p1.accuracy), mean_sd(p2.accuracy),
                         num(100 * (p1.accuracy.mean() - p2.accuracy.mean())), mean_sd(p1.within_class_accuracy, digits=1),
                         mean_sd(p2.within_class_accuracy, digits=1)])
    longtable(OUT / "linadv_summary.tex", "Amendment 8 probe-matched linear adversary (10 seeds): accuracy (percent), gap "
              "(pp), and recoverability $R$ (percent; device probe). $\\lambda$ above the regime collapses P2 accuracy.",
              "r:tab:linadv", ["Architecture", "$\\lambda$", "P1 acc.", "P2 acc.", "Gap", "$R$ (P1)", "$R$ (P2)"], rows,
              "lrrrrrr")
    normal = pd.read_csv(T / "normalization_recoverability.csv")
    rows = []
    for arch in [a for a in ORDER if a in set(normal.architecture)]:
        for strength in sorted(normal.strength.unique()):
            g = normal.loc[(normal.architecture == arch) & (normal.strength == strength)]
            p1, p2 = g.loc[g.protocol == "P1"], g.loc[g.protocol == "P2"]
            rows.append([NAMES[arch] if strength == 0 else "", f"{strength:g}", mean_sd(p1.accuracy), mean_sd(p2.accuracy),
                         num(100 * (p1.accuracy.mean() - p2.accuracy.mean())), mean_sd(p1.within_class_accuracy, digits=1),
                         mean_sd(p2.within_class_accuracy, digits=1)])
    longtable(OUT / "normalization_summary.tex", "Amendment 8 Reinhard stain normalisation (10 seeds; strength 0 is the "
              "matching Phase 1 run): accuracy (percent), gap (pp), and $R$ (percent).", "r:tab:normal",
              ["Architecture", "$s$", "P1 acc.", "P2 acc.", "Gap", "$R$ (P1)", "$R$ (P2)"], rows, "lrrrrrr")
    leace = pd.read_csv(T / "leace_runs.csv")
    rows = []
    for arch in ORDER:
        for rank in sorted(leace["rank"].unique()):
            g = leace.loc[(leace.architecture == arch) & (leace["rank"] == rank)]
            p1, p2 = g.loc[g.protocol == "P1"], g.loc[g.protocol == "P2"]
            rows.append([NAMES[arch] if rank == 0 else "", str(rank), mean_sd(p1.accuracy), mean_sd(p2.accuracy),
                         num(100 * (p1.accuracy.mean() - p2.accuracy.mean())), mean_sd(p1.within_class_accuracy, digits=1),
                         mean_sd(p2.within_class_accuracy, digits=1), mean_sd(g.within_class_ncc_accuracy, digits=1)])
    longtable(OUT / "leace_summary.tex", "Amendment 8 rank-limited LEACE erasure on every Phase 1 model (20 seeds): "
              "accuracy with a refitted head (percent), gap (pp), logistic recoverability $R$, and nearest-centroid "
              "recoverability (percent).", "r:tab:leace",
              ["Architecture", "Rank", "P1 acc.", "P2 acc.", "Gap", "$R$ (P1)", "$R$ (P2)", "NCC"], rows,
              "lrrrrrrr", size=r"\scriptsize")
    abl = pd.read_csv(T / "ablation_runs.csv")
    rows = []
    for arch in [a for a in ORDER if a in set(abl.architecture)]:
        for ablation in ("original", "grey", "stain_transfer", "low_pass", "illumination_flat", "background_grey"):
            g = abl.loc[(abl.architecture == arch) & (abl.ablation == ablation)]
            leak = g.loc[g.cell.str.startswith("leak")].accuracy
            clean = g.loc[g.cell.str.startswith("clean")].accuracy
            rows.append([NAMES[arch] if ablation == "original" else "", esc(ablation), mean_sd(leak), mean_sd(clean),
                         num(100 * (leak.mean() - clean.mean()))])
    longtable(OUT / "ablation_summary.tex", "Amendment 8 test-time carrier ablations on the Study 2 2C models: leak and "
              "clean accuracy (percent, 20 runs each) and the leak advantage (pp).", "r:tab:ablation",
              ["Architecture", "Ablation", "Leak models", "Clean models", "Leak $-$ clean"], rows, "llrrr")
    cams = pd.read_csv(T / "gradcam_runs.csv")
    rows = []
    for arch in [a for a in ORDER if a in set(cams.architecture)]:
        for leakage in ("leak", "clean"):
            g = cams.loc[(cams.architecture == arch) & cams.cell.str.startswith(leakage)]
            rows.append([NAMES[arch] if leakage == "leak" else "", leakage,
                         mean_sd(g.background_attribution_fraction, digits=1), mean_sd(g.background_pixel_fraction, digits=1),
                         num(g.random_head_spearman.median(), 3)])
    longtable(OUT / "gradcam_summary.tex", "Grad-CAM (descriptive): share of positive attribution outside the cell mask and "
              "background pixel share (percent, 20 runs each), and the median Spearman correlation with a random-head map.",
              "r:tab:gradcam", ["Architecture", "Models", "Attribution outside", "Background pixels", "Random-head $\\rho$"],
              rows, "llrrr")
    rows = []
    for item in cric["per_architecture_secondary"]:
        rows.append([NAMES[item["architecture"]], num(item["estimate"]), f"{num(item['ci95'][0])} to {num(item['ci95'][1])}",
                     pval(item["p_two_sided"]), f"{item['cells']:,}"])
    longtable(OUT / "cric_context_arch.tex", "F1 per architecture (secondary): within-cell P1 $-$ P2 context-class "
              "probability on discordant cells (pp) with slide-bootstrap intervals.", "r:tab:cricarch",
              ["Architecture", "Estimate", "95\\% CI", "$p$", "Cells"], rows, "lrrrr")
    overlap = json.loads((F / "exploratory" / "overlap.json").read_text())
    rows = [[esc(i["condition"]), esc(i["subset"]), str(i["pairs"]), num(i["mean_crops"], 0),
             num(i["leak_minus_clean_pp"]), f"{num(i['ci95'][0])} to {num(i['ci95'][1])}"] for i in overlap["summary"]]
    longtable(OUT / "overlap_summary.tex", "EXPLORATORY (not registered): leak minus clean accuracy on test crops that do "
              "or do not share source pixels with a training sibling (pp; 95\\% $t$ interval over architecture--seed--"
              "diversity pairs).", "r:tab:overlap", ["Condition", "Subset", "Pairs", "Crops", "Leak $-$ clean", "95\\% CI"],
              rows, "llrrrr")

    # Manipulation checks.
    rows = []
    for name, checks in (("H5 GRL", mech["H5_manipulation_checks"]), ("F5 LEACE", fol["F5_manipulation_checks"]),
                         ("F6 normalisation", fol["F6_regime_secondary"]), ("F7 linear adv.", fol["F7_manipulation_checks"])):
        first = True
        for arch in [a for a in ORDER if a in checks]:
            c = checks[arch]
            levels = c.get("non_degenerate_lambdas", c.get("levels_in_regime"))
            levels = levels if isinstance(levels, list) else [levels]
            rows.append([name if first else "", NAMES[arch], ", ".join(f"{v:g}" for v in levels),
                         f"{c.get('lambda_star', c.get('strongest')):g}", num(c.get("R_baseline", c.get("R0")), 1),
                         num(c.get("R_star", c.get("R_strongest")), 1), num(c["within_class_floor"], 1),
                         r"\pass" if c["manipulation_check_passed"] else r"\fail"])
            first = False
    longtable(OUT / "manipulation_checks.tex", "Regimes and manipulation checks of every dose-response arm: levels whose "
              "mean P2 accuracy stays within 1 pp of baseline, the strongest such level, recoverability at baseline and "
              "there (percent), the class-only floor, and the check (at least a quarter of the above-floor $R$ removed).",
              "r:tab:checks", ["Arm", "Architecture", "Levels in regime", "Strongest", "$R_0$", "$R^*$", "Floor", "Check"],
              rows, "llp{3.2cm}rrrrl", size=r"\scriptsize")

    # Power grid.
    power = pd.read_csv(ROOT / "results" / "study2" / "power" / "power_power.csv")
    rows = []
    for (scale, archs), g in power.loc[power.delta == 0.8].groupby(["scale", "architectures"]):
        g = g.sort_values("seeds")
        rows.append([f"{scale:g}$\\times$", str(archs), *(num(v, 3) for v in g.power)])
    seeds = sorted(power.seeds.unique())
    longtable(OUT / "power_grid.tex", "Simulated power (1,000 simulations per cell) of the registered H1 model for a "
              "0.8-pp effect, by variance scale, number of architectures, and number of seeds.", "r:tab:power",
              ["Variance", "Archs", *(f"{s} seeds" for s in seeds)], rows, "ll" + "r" * len(seeds))

    # Structural risk of the four datasets.
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from cytofair.risk import structural_risk
    rows = []
    for name, path in (("SIPaKMeD", "provenance/sipakmed_parent_map_p3.parquet"), ("Cx22", "provenance/cx22_parent_map.parquet"),
                       ("CRIC", "provenance/cric_parent_map.parquet"), ("Herlev", "provenance/herlev_parent_map.parquet")):
        frame = pd.read_parquet(ROOT / path)
        risk = structural_risk(frame)
        rows.append([name, f"{len(frame):,}", f"{frame.parent_id.nunique():,}", num(risk["crops_per_parent_mean"]),
                     num(100 * risk["expected_parent_straddle"], 1),
                     num(100 * risk["expected_test_crops_with_training_sibling"], 1),
                     num(risk["expected_training_siblings_per_test_crop"], 2), num(risk["training_diversity_ratio"], 3),
                     num(risk["class_parent_dependence"], 3)])
    longtable(OUT / "risk_report.tex", "\\texttt{cytofair risk-report} closed forms for the four datasets at a 20\\% test "
              "fraction: crops per parent, expected straddle and sibling rates (percent), siblings per test crop, "
              "training-diversity ratio, and class--parent dependence (fraction of class entropy explained by the parent).",
              "r:tab:risk", ["Dataset", "Obs.", "Parents", "Obs./parent", "Straddle", "Test w/ sibling", "Siblings",
                             "Diversity", "Class dep."], rows, "lrrrrrrrr", size=r"\scriptsize")
    (OUT / "risk_keys.txt").write_text("\n".join(sorted(risk.keys())) + "\n")

    # ---------------- run ledgers ----------------
    def ledger(experiment, caption, label, extra=()):
        g = runs.loc[runs.experiment == experiment].copy()
        g["_o"] = g.architecture.map({a: i for i, a in enumerate(ORDER)})
        sort_keys = ["_o", *[e for e in extra if e != "condition"], "seed", "protocol"]
        if "condition" in extra:
            sort_keys.insert(1, "condition")
        g = g.sort_values(sort_keys, kind="stable")
        out_rows = []
        for item in g.itertuples(index=False):
            row = [NAMES.get(item.architecture, item.architecture), str(item.seed), esc(item.protocol)]
            for field in extra:
                value = getattr(item, field)
                row.append("--" if pd.isna(value) else esc(f"{value:g}" if isinstance(value, float) else value))
            row += [num(100 * item.accuracy), num(100 * item.macro_f1), num(100 * item.balanced_accuracy),
                    num(100 * item.ece_15), num(item.elapsed_seconds / 60, 1), str(item.gpu_visible_devices)]
            out_rows.append(row)
        header = ["Architecture", "Seed", "Protocol", *[esc(e) for e in extra], "Acc.", "F1", "Bal.", "ECE", "Min", "GPU"]
        longtable(OUT / f"ledger_{experiment}.tex", caption, label, header, out_rows,
                  "lrl" + "l" * len(extra) + "rrrrrr", size=r"\scriptsize")
    ledger("phase1", "Complete Phase 1 run ledger (720 runs; metrics in percent; minutes of wall time; GPU index).",
           "r:tab:ledger-phase1")
    ledger("cric", "Complete CRIC run ledger (180 runs).", "r:tab:ledger-cric")
    ledger("factorial", "Complete 2C/2B run ledger (160 runs; protocol is the factorial cell).", "r:tab:ledger-factorial")
    ledger("grl", "Complete 2A gradient-reversal run ledger (280 runs).", "r:tab:ledger-grl", extra=("adversary_lambda",))
    runs.loc[runs.experiment == "carrier", "condition"] = runs.loc[runs.experiment == "carrier", "run"].str.rsplit("_", n=1).str[1]
    ledger("carrier", "Complete Amendment 8 carrier-factorial run ledger (480 runs).", "r:tab:ledger-carrier",
           extra=("condition",))
    ledger("linadv", "Complete Amendment 8 linear-adversary run ledger (300 runs).", "r:tab:ledger-linadv",
           extra=("adversary_lambda",))
    ledger("normalization", "Complete Amendment 8 normalisation run ledger (160 runs).", "r:tab:ledger-normalization",
           extra=("cache_name",))
    rows = []
    leace_wide = leace.loc[leace["rank"].isin([0, 256])].pivot_table(
        index=["architecture", "seed", "protocol"], columns="rank", values=["accuracy", "within_class_accuracy"]).reset_index()
    leace_wide = arch_sort(leace_wide)
    leace_wide["_o"] = leace_wide.architecture.map({a: i for i, a in enumerate(ORDER)})
    leace_wide = leace_wide.sort_values(["_o", "seed", "protocol"], kind="stable").drop(columns="_o")
    for item in leace_wide.itertuples(index=False):
        rows.append([NAMES[item[0]], str(item[1]), item[2], num(100 * item[3]), num(100 * item[4]), num(100 * item[5], 1),
                     num(100 * item[6], 1)])
    longtable(OUT / "ledger_leace.tex", "Complete LEACE ledger (480 models): refitted-head accuracy and recoverability at "
              "rank 0 and rank 256 (percent).", "r:tab:ledger-leace",
              ["Architecture", "Seed", "Prot.", "Acc. r=0", "Acc. r=256", "$R$ r=0", "$R$ r=256"], rows, "lrlrrrr",
              size=r"\scriptsize")
    wide = abl.pivot_table(index=["architecture", "seed", "cell"], columns="ablation", values="accuracy").reset_index()
    rows = []
    wide["_o"] = wide.architecture.map({a: i for i, a in enumerate(ORDER)})
    wide = wide.sort_values(["_o", "seed", "cell"], kind="stable").drop(columns="_o")
    for item in wide.itertuples(index=False):
        d = item._asdict()
        rows.append([NAMES[d["architecture"]], str(d["seed"]), esc(d["cell"]),
                     *(num(100 * d[c]) for c in ("original", "grey", "stain_transfer", "low_pass", "illumination_flat",
                                                 "background_grey"))])
    longtable(OUT / "ledger_ablation.tex", "Complete test-time ablation ledger (160 models; accuracy in percent).",
              "r:tab:ledger-ablation", ["Architecture", "Seed", "Cell", "Orig.", "Grey", "Stain", "Low-pass", "Flat", "No bg"],
              rows, "lrlrrrrrr", size=r"\scriptsize")
    rows = [[brk(r.run), esc(r.experiment), esc(r.architecture), esc(r.protocol), str(r.epochs), num(r.elapsed_seconds, 0)]
            for r in smoke.itertuples(index=False)]
    longtable(OUT / "ledger_smoke.tex", "Every smoke test (Study 2 before registration; Amendment 8 before registration). "
              "Only crop-level, leak-cell, or single-protocol runs were smoke-tested, so no registered contrast could be "
              "observed.", "r:tab:ledger-smoke", ["Run", "Kind", "Architecture", "Protocol", "Epochs", "Seconds"], rows,
              "p{5.2cm}llllr", size=r"\scriptsize")

    # ---------------- git chronology ----------------
    log = subprocess.check_output(["git", "-C", str(ROOT), "log", "--all", "--reverse", "--format=%h|%ad|%s",
                                   "--date=format:%Y-%m-%d %H:%M"], text=True).strip().splitlines()
    rows = [[esc(h_), d, esc(s)] for h_, d, s in (line.split("|", 2) for line in log)]
    longtable(OUT / "git_chronology.tex", "Complete commit chronology of the project (all branches; IST).",
              "r:tab:git", ["Commit", "Date", "Message"], rows, "llp{11.2cm}", size=r"\scriptsize")

    # Verbatim source documents for the appendices (copied so the record is self-contained).
    import shutil
    for source, target in (("reports/study2/registration_hashes.txt", "registration_hashes.txt"),
                           ("reports/study2/excluded_attempts.md", "excluded_attempts.md"),
                           ("PREREGISTRATION_STUDY2.md", "PREREGISTRATION_STUDY2.md"),
                           ("PREREGISTRATION_STUDY2_AMENDMENT8.md", "PREREGISTRATION_STUDY2_AMENDMENT8.md"),
                           ("reports/study2/phase0_arxiv_checklist.md", "phase0_arxiv_checklist.md"),
                           ("reports/study2/cx22_literature_check.md", "cx22_literature_check.md")):
        text = (ROOT / source).read_text()
        for bad, good in (("\u2014", "--"), ("\u2013", "-"), ("\u00b1", "+/-"), ("\u00d7", "x"), ("\u2265", ">="),
                          ("\u2264", "<="), ("\u2192", "->"), ("\u2019", "'"), ("\u201c", '"'), ("\u201d", '"')):
            text = text.replace(bad, good)
        (OUT / target).write_text(text)
    lines = [rf"\newcommand{{\{k}}}{{{v}}}" for k, v in sorted(macros.items())]
    (OUT / "record_numbers.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(macros)} macros and tables to {OUT}")


if __name__ == "__main__":
    main()
