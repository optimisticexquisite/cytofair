"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .classification import train_classifier
from .constants import SIPAKMED_CROP_COUNTS, SIPAKMED_PARENT_COUNTS, SIPAKMED_TOTAL_CROPS
from .cric import recover_classifications
from .cx22 import recover_all as recover_cx22
from .cx22 import subset_parent_overlap
from .cx22_dataset import materialize_dataset
from .download import download_manifest
from .duplicates import (
    attach_p3_groups,
    duplicate_pairs,
    hash_manifest,
    synthetic_detector_check,
)
from .extract import extract_manifest
from .herlev import recover_one_cell_per_image
from .probe import summarize_probe, train_parent_probe
from .provenance import read_map, validate_provenance, write_map
from .sipakmed import (
    bbox_iou_pairs,
    cross_split_pixel_overlap,
    recover_branch_a,
    recover_branch_b,
)
from .splits import export_protocol_splits, make_grouped, make_p1
from .table1 import structural_table
from .verification import git_preregistration_evidence


def _validate_sipakmed_maps(
    branch_a,
    branch_b,
    *,
    validation_path: str | Path,
    overlap_path: str | Path,
) -> None:
    """Apply only the preregistered provenance gates and persist audit evidence."""
    validation = validate_provenance(
        branch_b,
        expected_total=SIPAKMED_TOTAL_CROPS,
        expected_parent_counts=SIPAKMED_PARENT_COUNTS,
        expected_class_crop_counts=SIPAKMED_CROP_COUNTS,
        branch_a=branch_a,
        branch_b=branch_b,
        require_bbox=True,
    )
    Path(validation_path).parent.mkdir(parents=True, exist_ok=True)
    validation.frame().to_csv(validation_path, index=False)
    overlaps = bbox_iou_pairs(branch_b)
    Path(overlap_path).parent.mkdir(parents=True, exist_ok=True)
    overlaps.to_parquet(overlap_path, index=False)
    validation.raise_for_failure()


def _download(args: argparse.Namespace) -> None:
    paths = download_manifest(
        args.manifest, args.output, dataset=args.dataset, connections=args.connections
    )
    for path in paths:
        print(path)


def _extract(args: argparse.Namespace) -> None:
    paths = extract_manifest(
        args.manifest, args.raw, args.output, dataset=args.dataset
    )
    for path in paths:
        print(path)


def _sipak(args: argparse.Namespace) -> None:
    branch_a = recover_branch_a(args.root)
    checkpoint = Path(args.branch_b).with_suffix(".partial.parquet")
    branch_b = recover_branch_b(
        branch_a, workers=args.workers, checkpoint_path=checkpoint
    )
    write_map(branch_a, args.branch_a)
    write_map(branch_b, args.branch_b)
    write_map(branch_a, args.output)
    _validate_sipakmed_maps(
        branch_a,
        branch_b,
        validation_path=args.validation,
        overlap_path=args.overlap,
    )
    checkpoint.unlink(missing_ok=True)


def _sipak_a(args: argparse.Namespace) -> None:
    branch_a = recover_branch_a(args.root)
    validation = validate_provenance(
        branch_a,
        expected_total=SIPAKMED_TOTAL_CROPS,
        expected_parent_counts=SIPAKMED_PARENT_COUNTS,
        expected_class_crop_counts=SIPAKMED_CROP_COUNTS,
    )
    write_map(branch_a, args.output)
    Path(args.validation).parent.mkdir(parents=True, exist_ok=True)
    validation.frame().to_csv(args.validation, index=False)
    validation.raise_for_failure()
    print(f"SIPaKMeD crops={len(branch_a)} parents={branch_a.parent_id.nunique()}")


def _sipak_b(args: argparse.Namespace) -> None:
    branch_a = read_map(args.branch_a)
    checkpoint = Path(args.output).with_suffix(".partial.parquet")
    branch_b = recover_branch_b(
        branch_a, workers=args.workers, checkpoint_path=checkpoint
    )
    write_map(branch_b, args.output)
    _validate_sipakmed_maps(
        branch_a,
        branch_b,
        validation_path=args.validation,
        overlap_path=args.overlap,
    )
    checkpoint.unlink(missing_ok=True)


def _validate_sipak(args: argparse.Namespace) -> None:
    branch_a = read_map(args.branch_a)
    branch_b = read_map(args.branch_b)
    _validate_sipakmed_maps(
        branch_a,
        branch_b,
        validation_path=args.validation,
        overlap_path=args.overlap,
    )
    write_map(branch_a, args.output)
    print(
        f"SIPaKMeD provenance accepted: crops={len(branch_b)} "
        f"parents={branch_b.parent_id.nunique()}"
    )


def _cx22(args: argparse.Namespace) -> None:
    frame = recover_cx22(args.root, args.image_root)
    if len(frame) != 1320:
        raise ValueError(f"Expected 1320 Cx22 crops; found {len(frame)}")
    if args.image_root:
        validation = validate_provenance(frame, expected_total=1320, require_bbox=True)
        Path(args.validation).parent.mkdir(parents=True, exist_ok=True)
        validation.frame().to_csv(args.validation, index=False)
        validation.raise_for_failure()
    write_map(frame, args.output)
    overlap = subset_parent_overlap(frame)
    Path(args.overlap).parent.mkdir(parents=True, exist_ok=True)
    overlap.to_csv(args.overlap, index=False)
    print(overlap.to_string(index=False))


def _materialize_cx22(args: argparse.Namespace) -> None:
    frame = materialize_dataset(
        read_map(args.map),
        args.labels,
        args.assets,
        overwrite=args.overwrite,
    )
    write_map(frame, args.output)
    audit = {
        "crops": len(frame),
        "parents": int(frame.parent_id.nunique()),
        "cytoplasm_instances": int(frame.n_cyto_instances.sum()),
        "nucleus_instances": int(frame.n_nuc_instances.sum()),
        "subsets": {str(key): int(value) for key, value in frame.subset.value_counts().items()},
    }
    Path(args.audit).parent.mkdir(parents=True, exist_ok=True)
    Path(args.audit).write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


def _cric(args: argparse.Namespace) -> None:
    frame = recover_classifications(args.classifications)
    write_map(frame, args.output)
    print(f"CRIC crops={len(frame)} parents={frame.parent_id.nunique()}")


def _herlev(args: argparse.Namespace) -> None:
    frame = recover_one_cell_per_image(args.root)
    if len(frame) != args.expected_total:
        raise ValueError(f"Expected {args.expected_total} Herlev cells; found {len(frame)}")
    write_map(frame, args.output)
    print(f"Herlev crops={len(frame)} parents={frame.parent_id.nunique()}")


def _table1(args: argparse.Namespace) -> None:
    maps = {}
    for item in args.map:
        dataset, path = item.split("=", 1)
        maps[dataset] = read_map(path)
    summary, seeds = structural_table(maps)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    seeds.to_csv(args.seed_output, index=False)
    print(summary.to_string(index=False))


def _git_evidence(args: argparse.Namespace) -> None:
    print(json.dumps(git_preregistration_evidence(args.repository), indent=2))


def _make_splits(args: argparse.Namespace) -> None:
    from .constants import CONFIRMATORY_SEEDS

    manifest = export_protocol_splits(
        read_map(args.map), args.output, dataset=args.dataset, seeds=CONFIRMATORY_SEEDS
    )
    print(manifest.to_string(index=False))


def _pixel_overlap(args: argparse.Namespace) -> None:
    import pandas as pd

    from .constants import CONFIRMATORY_SEEDS

    provenance = read_map(args.map)
    rows = []
    for seed in CONFIRMATORY_SEEDS:
        splits = {
            "P1": make_p1(provenance, seed),
            "P2": make_grouped(provenance, seed, protocol="P2"),
            "P3": make_grouped(provenance, seed, protocol="P3"),
        }
        for protocol, split in splits.items():
            rows.append(
                {
                    "dataset": args.dataset,
                    "protocol": protocol,
                    "seed": seed,
                    **cross_split_pixel_overlap(provenance, split),
                }
            )
    seed_frame = pd.DataFrame(rows)
    metrics = [
        "overlapping_cross_split_pair_fraction",
        "test_crop_pixel_overlap_fraction",
    ]
    summary = (
        seed_frame.groupby(["dataset", "protocol"], observed=True)[metrics]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    for count_column in [column for column in summary if column.endswith("_count")]:
        if not (summary[count_column] >= 5).all():
            raise ValueError("Pixel-overlap summaries require at least five seeds")
    Path(args.seed_output).parent.mkdir(parents=True, exist_ok=True)
    seed_frame.to_csv(args.seed_output, index=False)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(summary.to_string(index=False))


def _train_probe(args: argparse.Namespace) -> None:
    result = train_parent_probe(
        read_map(args.map),
        seed=args.seed,
        output=args.output,
        image_mode=args.image_mode,
        shuffled_labels=args.shuffled_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        workers=args.workers,
        learning_rate=args.learning_rate,
        pretrained=not args.no_pretrained,
        repository=args.repository,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "history"}, indent=2))


def _summarize_probe(args: argparse.Namespace) -> None:
    summary, control = summarize_probe(args.results)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    Path(args.control_output).write_text(json.dumps(control, indent=2))
    print(summary.to_string(index=False))
    print(json.dumps(control, indent=2))


def _train_classifier(args: argparse.Namespace) -> None:
    result = train_classifier(
        read_map(args.map),
        dataset=args.dataset,
        protocol=args.protocol,
        architecture=args.architecture,
        seed=args.seed,
        output=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        workers=args.workers,
        learning_rate=args.learning_rate,
        pretrained=not args.no_pretrained,
        repository=args.repository,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "history"}, indent=2))


def _train_semantic_segmentation(args: argparse.Namespace) -> None:
    from .segmentation import train_semantic_segmentation

    result = train_semantic_segmentation(
        read_map(args.map),
        protocol=args.protocol,
        architecture=args.architecture,
        seed=args.seed,
        output=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        workers=args.workers,
        learning_rate=args.learning_rate,
        image_size=args.image_size,
        pretrained=not args.no_pretrained,
        repository=args.repository,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "history"}, indent=2))


def _train_instance_segmentation(args: argparse.Namespace) -> None:
    from .instance_segmentation import train_instance_segmentation

    result = train_instance_segmentation(
        read_map(args.map),
        protocol=args.protocol,
        architecture=args.architecture,
        seed=args.seed,
        output=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        workers=args.workers,
        learning_rate=args.learning_rate,
        image_size=args.image_size,
        pretrained=not args.no_pretrained,
        repository=args.repository,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "history"}, indent=2))


def _summarize_segmentation(args: argparse.Namespace) -> None:
    from .segmentation_results import summarize_segmentation

    frames = summarize_segmentation(args.results)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(output / f"cx22_segmentation_{name}.csv", index=False)
    print(frames["summary"].to_string(index=False))
    print(frames["primary"].to_string(index=False))


def _summarize_classification(args: argparse.Namespace) -> None:
    from .results import summarize_classification

    summary, primary, secondary, ordering, ranking = summarize_classification(args.results)
    outputs = {
        args.output: summary,
        args.contrasts: primary,
        args.secondary: secondary,
        args.ordering: ordering,
        args.ranking: ranking,
    }
    for path, frame in outputs.items():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    print(summary.to_string(index=False))
    print(primary.to_string(index=False))


def _duplicates(args: argparse.Namespace) -> None:
    provenance = read_map(args.map)
    hashes = hash_manifest(provenance)
    pairs = duplicate_pairs(
        hashes, phash_threshold=args.phash_threshold, dhash_threshold=args.dhash_threshold
    )
    Path(args.hashes).parent.mkdir(parents=True, exist_ok=True)
    hashes.to_parquet(args.hashes, index=False)
    pairs.to_csv(args.pairs, index=False)
    write_map(attach_p3_groups(provenance, pairs), args.output)
    print(
        json.dumps(
            {
                "crops": len(hashes),
                "consensus_pairs": len(pairs),
                "cross_parent_pairs": int(pairs.cross_parent.sum()) if not pairs.empty else 0,
            },
            indent=2,
        )
    )


def _validate_duplicate_detector(args: argparse.Namespace) -> None:
    result = synthetic_detector_check(
        args.base,
        args.output_dir,
        negative_paths=args.negative,
        phash_threshold=args.phash_threshold,
        dhash_threshold=args.dhash_threshold,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cytofair")
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser("download", help="Download checksum-declared raw sources")
    command.add_argument("--manifest", default="data_sources.json")
    command.add_argument("--output", default="data/raw")
    command.add_argument("--dataset")
    command.add_argument("--connections", type=int, default=3, choices=(1, 2, 3))
    command.set_defaults(function=_download)

    command = commands.add_parser("extract", help="Extract downloaded dataset archives")
    command.add_argument("--manifest", default="data_sources.json")
    command.add_argument("--raw", default="data/raw")
    command.add_argument("--output", default="data/extracted")
    command.add_argument("--dataset")
    command.set_defaults(function=_extract)

    command = commands.add_parser("recover-sipakmed", help="Run SIPaKMeD Branch A and B")
    command.add_argument("root")
    command.add_argument("--workers", type=int, default=1)
    command.add_argument("--branch-a", default="provenance/sipakmed_branch_a.parquet")
    command.add_argument("--branch-b", default="provenance/sipakmed_branch_b.parquet")
    command.add_argument("--output", default="provenance/sipakmed_parent_map.parquet")
    command.add_argument("--validation", default="reports/generated/sipakmed_validation.csv")
    command.add_argument("--overlap", default="reports/generated/sipakmed_bbox_overlap.parquet")
    command.set_defaults(function=_sipak)

    command = commands.add_parser("recover-sipakmed-a", help="Run filename provenance only")
    command.add_argument("root")
    command.add_argument("--output", default="provenance/sipakmed_branch_a.parquet")
    command.add_argument("--validation", default="reports/generated/sipakmed_branch_a_validation.csv")
    command.set_defaults(function=_sipak_a)

    command = commands.add_parser("recover-sipakmed-b", help="Run exhaustive image provenance")
    command.add_argument("branch_a")
    command.add_argument("--workers", type=int, default=1)
    command.add_argument("--output", default="provenance/sipakmed_branch_b.parquet")
    command.add_argument("--validation", default="reports/generated/sipakmed_validation.csv")
    command.add_argument("--overlap", default="reports/generated/sipakmed_bbox_overlap.parquet")
    command.set_defaults(function=_sipak_b)

    command = commands.add_parser(
        "validate-sipakmed", help="Validate completed SIPaKMeD Branch A/B maps"
    )
    command.add_argument("branch_a")
    command.add_argument("branch_b")
    command.add_argument("--output", default="provenance/sipakmed_parent_map.parquet")
    command.add_argument("--validation", default="reports/generated/sipakmed_validation.csv")
    command.add_argument("--overlap", default="reports/generated/sipakmed_bbox_iou.parquet")
    command.set_defaults(function=_validate_sipak)

    command = commands.add_parser("recover-cx22", help="Read source names and ROIs from Cx22")
    command.add_argument("root")
    command.add_argument("--image-root", help="extracted CCEDD parent PNG directory")
    command.add_argument("--output", default="provenance/cx22_parent_map.parquet")
    command.add_argument("--overlap", default="reports/generated/cx22_subset_overlap.csv")
    command.add_argument("--validation", default="reports/generated/cx22_validation.csv")
    command.set_defaults(function=_cx22)

    command = commands.add_parser(
        "materialize-cx22", help="Reconstruct Cx22 RGB ROIs and decode released masks"
    )
    command.add_argument("map", help="validated Cx22 provenance map")
    command.add_argument("--labels", required=True, help="root containing the three Cx22 releases")
    command.add_argument("--assets", required=True, help="generated RGB/mask asset directory")
    command.add_argument("--output", default="provenance/cx22_model_map.parquet")
    command.add_argument("--audit", default="reports/generated/cx22_materialization.json")
    command.add_argument("--overwrite", action="store_true")
    command.set_defaults(function=_materialize_cx22)

    command = commands.add_parser("recover-cric", help="Read CRIC published cell coordinates")
    command.add_argument("classifications")
    command.add_argument("--output", default="provenance/cric_parent_map.parquet")
    command.set_defaults(function=_cric)

    command = commands.add_parser("recover-herlev", help="Build Herlev one-image parent map")
    command.add_argument("root")
    command.add_argument("--expected-total", type=int, default=917)
    command.add_argument("--output", default="provenance/herlev_parent_map.parquet")
    command.set_defaults(function=_herlev)

    command = commands.add_parser("table1", help="Generate structural leakage Table 1")
    command.add_argument(
        "--map", action="append", required=True, metavar="DATASET=PATH", help="canonical map"
    )
    command.add_argument("--output", default="results/table1_structural_leakage.csv")
    command.add_argument("--seed-output", default="results/table1_seed_level.csv")
    command.set_defaults(function=_table1)

    command = commands.add_parser("git-evidence", help="Verify preregistration chronology")
    command.add_argument("repository", nargs="?", default=".")
    command.set_defaults(function=_git_evidence)

    command = commands.add_parser("make-splits", help="Export fixed P1/P1-prime/P2/P3 assets")
    command.add_argument("map")
    command.add_argument("--dataset", required=True)
    command.add_argument("--output", default="splits")
    command.set_defaults(function=_make_splits)

    command = commands.add_parser(
        "pixel-overlap", help="Measure literal bbox overlap across split boundaries"
    )
    command.add_argument("map", help="Branch-B map containing recovered bounding boxes")
    command.add_argument("--dataset", default="SIPaKMeD")
    command.add_argument("--output", default="results/pixel_overlap_summary.csv")
    command.add_argument("--seed-output", default="results/pixel_overlap_seed_level.csv")
    command.set_defaults(function=_pixel_overlap)

    command = commands.add_parser("train-probe", help="Train one parent-identity probe run")
    command.add_argument("map")
    command.add_argument("--seed", type=int, required=True)
    command.add_argument("--output", required=True)
    command.add_argument(
        "--image-mode",
        choices=("rgb", "grayscale", "stain_normalized", "background_only", "highpass"),
        default="rgb",
    )
    command.add_argument("--shuffled-labels", action="store_true")
    command.add_argument("--epochs", type=int, default=15)
    command.add_argument("--batch-size", type=int, default=64)
    command.add_argument("--workers", type=int, default=2)
    command.add_argument("--learning-rate", type=float, default=1e-4)
    command.add_argument("--no-pretrained", action="store_true")
    command.add_argument("--repository", default=".")
    command.set_defaults(function=_train_probe)

    command = commands.add_parser("summarize-probe", help="Summarize probe seeds and control")
    command.add_argument("results", nargs="+")
    command.add_argument("--output", default="results/parent_probe_summary.csv")
    command.add_argument("--control-output", default="results/parent_probe_shuffled_control.json")
    command.set_defaults(function=_summarize_probe)

    command = commands.add_parser("train-classifier", help="Train one matched protocol run")
    command.add_argument("map")
    command.add_argument("--dataset", required=True)
    command.add_argument("--protocol", choices=("P1", "P1-prime", "P2", "P3"), required=True)
    command.add_argument(
        "--architecture",
        choices=(
            "resnet18",
            "resnet50",
            "densenet121",
            "efficientnet_b0",
            "mobilenet_v2",
            "vit_b_16",
        ),
        required=True,
    )
    command.add_argument("--seed", type=int, required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--epochs", type=int, default=20)
    command.add_argument("--batch-size", type=int, default=64)
    command.add_argument("--workers", type=int, default=2)
    command.add_argument("--learning-rate", type=float, default=1e-4)
    command.add_argument("--no-pretrained", action="store_true")
    command.add_argument("--repository", default=".")
    command.set_defaults(function=_train_classifier)

    command = commands.add_parser(
        "train-semantic-segmentation", help="Train one resumable Cx22 semantic run"
    )
    command.add_argument("map")
    command.add_argument(
        "--protocol", choices=("P1", "P1-prime", "P2", "P3", "OFFICIAL"), required=True
    )
    command.add_argument(
        "--architecture",
        choices=(
            "unet_resnet34",
            "unetplusplus_resnet34",
            "deeplabv3plus_resnet50",
            "fpn_resnet34",
            "manet_resnet34",
        ),
        required=True,
    )
    command.add_argument("--seed", type=int, required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--epochs", type=int, default=30)
    command.add_argument("--batch-size", type=int, default=16)
    command.add_argument("--workers", type=int, default=2)
    command.add_argument("--learning-rate", type=float, default=1e-4)
    command.add_argument("--image-size", type=int, default=256)
    command.add_argument("--no-pretrained", action="store_true")
    command.add_argument("--repository", default=".")
    command.set_defaults(function=_train_semantic_segmentation)

    command = commands.add_parser(
        "train-instance-segmentation", help="Train one resumable Cx22 Mask R-CNN run"
    )
    command.add_argument("map")
    command.add_argument(
        "--protocol", choices=("P1", "P1-prime", "P2", "P3", "OFFICIAL"), required=True
    )
    command.add_argument(
        "--architecture",
        choices=("maskrcnn_resnet50_fpn", "maskrcnn_resnet50_fpn_v2"),
        required=True,
    )
    command.add_argument("--seed", type=int, required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--epochs", type=int, default=30)
    command.add_argument("--batch-size", type=int, default=2)
    command.add_argument("--workers", type=int, default=2)
    command.add_argument("--learning-rate", type=float, default=1e-4)
    command.add_argument("--image-size", type=int, default=256)
    command.add_argument("--no-pretrained", action="store_true")
    command.add_argument("--repository", default=".")
    command.set_defaults(function=_train_instance_segmentation)

    command = commands.add_parser(
        "summarize-segmentation", help="Summarize Cx22 segmentation seeds and inference"
    )
    command.add_argument("results", nargs="+")
    command.add_argument("--output", default="results/cx22_segmentation_summary")
    command.set_defaults(function=_summarize_segmentation)

    command = commands.add_parser(
        "summarize-classification", help="Summarize seeds, effects, BH family, and ordering"
    )
    command.add_argument("results", nargs="+")
    command.add_argument("--output", default="results/classification_summary.csv")
    command.add_argument("--contrasts", default="results/classification_primary_contrasts.csv")
    command.add_argument("--secondary", default="results/classification_secondary_contrasts.csv")
    command.add_argument("--ordering", default="results/classification_ordering.csv")
    command.add_argument("--ranking", default="results/classification_ranking_stability.csv")
    command.set_defaults(function=_summarize_classification)

    command = commands.add_parser("audit-duplicates", help="Hash crops and create P3 groups")
    command.add_argument("map")
    command.add_argument("--hashes", default="duplicates/hash_manifest.parquet")
    command.add_argument("--pairs", default="duplicates/near_duplicate_manifest.csv")
    command.add_argument("--output", default="provenance/parent_map_p3.parquet")
    command.add_argument("--phash-threshold", type=int, default=6)
    command.add_argument("--dhash-threshold", type=int, default=6)
    command.set_defaults(function=_duplicates)

    command = commands.add_parser(
        "validate-duplicate-detector", help="Score hash precision/recall on injected copies"
    )
    command.add_argument("base")
    command.add_argument("--negative", action="append", default=[])
    command.add_argument("--output-dir", default="results/duplicates/synthetic_check")
    command.add_argument(
        "--output", default="results/duplicates/synthetic_detector_check.json"
    )
    command.add_argument("--phash-threshold", type=int, default=6)
    command.add_argument("--dhash-threshold", type=int, default=6)
    command.set_defaults(function=_validate_duplicate_detector)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
