"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .classification import train_classifier
from .constants import SIPAKMED_PARENT_COUNTS, SIPAKMED_TOTAL_CROPS
from .cric import recover_classifications
from .cx22 import recover_all as recover_cx22
from .cx22 import subset_parent_overlap
from .download import download_manifest
from .duplicates import attach_p3_groups, duplicate_pairs, hash_manifest
from .extract import extract_manifest
from .herlev import recover_one_cell_per_image
from .probe import summarize_probe, train_parent_probe
from .provenance import read_map, validate_provenance, write_map
from .sipakmed import bbox_iou_pairs, recover_branch_a, recover_branch_b
from .table1 import structural_table
from .verification import git_preregistration_evidence


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
    branch_b = recover_branch_b(branch_a, workers=args.workers)
    validation = validate_provenance(
        branch_b,
        expected_total=SIPAKMED_TOTAL_CROPS,
        expected_parent_counts=SIPAKMED_PARENT_COUNTS,
        branch_a=branch_a,
        branch_b=branch_b,
        min_accepted_fraction=0.95,
        require_bbox=True,
    )
    write_map(branch_a, args.branch_a)
    write_map(branch_b, args.branch_b)
    Path(args.validation).parent.mkdir(parents=True, exist_ok=True)
    validation.frame().to_csv(args.validation, index=False)
    overlaps = bbox_iou_pairs(branch_b)
    Path(args.overlap).parent.mkdir(parents=True, exist_ok=True)
    overlaps.to_parquet(args.overlap, index=False)
    validation.raise_for_failure()


def _sipak_a(args: argparse.Namespace) -> None:
    branch_a = recover_branch_a(args.root)
    validation = validate_provenance(
        branch_a,
        expected_total=SIPAKMED_TOTAL_CROPS,
        expected_parent_counts=SIPAKMED_PARENT_COUNTS,
    )
    write_map(branch_a, args.output)
    Path(args.validation).parent.mkdir(parents=True, exist_ok=True)
    validation.frame().to_csv(args.validation, index=False)
    validation.raise_for_failure()
    print(f"SIPaKMeD crops={len(branch_a)} parents={branch_a.parent_id.nunique()}")


def _sipak_b(args: argparse.Namespace) -> None:
    branch_a = read_map(args.branch_a)
    branch_b = recover_branch_b(branch_a, workers=args.workers)
    validation = validate_provenance(
        branch_b,
        expected_total=SIPAKMED_TOTAL_CROPS,
        expected_parent_counts=SIPAKMED_PARENT_COUNTS,
        branch_a=branch_a,
        branch_b=branch_b,
        min_accepted_fraction=0.95,
        require_bbox=True,
    )
    write_map(branch_b, args.output)
    Path(args.validation).parent.mkdir(parents=True, exist_ok=True)
    validation.frame().to_csv(args.validation, index=False)
    overlaps = bbox_iou_pairs(branch_b)
    Path(args.overlap).parent.mkdir(parents=True, exist_ok=True)
    overlaps.to_parquet(args.overlap, index=False)
    validation.raise_for_failure()


def _cx22(args: argparse.Namespace) -> None:
    frame = recover_cx22(args.root)
    write_map(frame, args.output)
    overlap = subset_parent_overlap(frame)
    Path(args.overlap).parent.mkdir(parents=True, exist_ok=True)
    overlap.to_csv(args.overlap, index=False)
    print(overlap.to_string(index=False))


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
    command.add_argument("--branch-b", default="provenance/sipakmed_parent_map.parquet")
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
    command.add_argument("--output", default="provenance/sipakmed_parent_map.parquet")
    command.add_argument("--validation", default="reports/generated/sipakmed_validation.csv")
    command.add_argument("--overlap", default="reports/generated/sipakmed_bbox_overlap.parquet")
    command.set_defaults(function=_sipak_b)

    command = commands.add_parser("recover-cx22", help="Read source names and ROIs from Cx22")
    command.add_argument("root")
    command.add_argument("--output", default="provenance/cx22_parent_map.parquet")
    command.add_argument("--overlap", default="reports/generated/cx22_subset_overlap.csv")
    command.set_defaults(function=_cx22)

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

    command = commands.add_parser("train-probe", help="Train one parent-identity probe run")
    command.add_argument("map")
    command.add_argument("--seed", type=int, required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--image-mode", choices=("rgb", "grayscale", "highpass"), default="rgb")
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

    command = commands.add_parser("audit-duplicates", help="Hash crops and create P3 groups")
    command.add_argument("map")
    command.add_argument("--hashes", default="duplicates/hash_manifest.parquet")
    command.add_argument("--pairs", default="duplicates/near_duplicate_manifest.csv")
    command.add_argument("--output", default="provenance/parent_map_p3.parquet")
    command.add_argument("--phash-threshold", type=int, default=6)
    command.add_argument("--dhash-threshold", type=int, default=6)
    command.set_defaults(function=_duplicates)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
