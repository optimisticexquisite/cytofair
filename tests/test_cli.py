from cytofair.cli import build_parser


def test_full_sipak_recovery_keeps_canonical_and_audit_outputs_separate():
    args = build_parser().parse_args(["recover-sipakmed", "/data"])
    assert args.output.endswith("sipakmed_parent_map.parquet")
    assert args.branch_b.endswith("sipakmed_branch_b.parquet")
    assert args.output != args.branch_b
