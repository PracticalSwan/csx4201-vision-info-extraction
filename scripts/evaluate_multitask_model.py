#!/usr/bin/env python3
"""Evaluate a trained public multi-task checkpoint on one explicit split/stream set."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config as cfgmod  # noqa: E402
from src.information_extraction.layoutxlm_data import (  # noqa: E402
    BIO_LABELS,
    load_model_examples,
)
from src.information_extraction.model_dataset import (  # noqa: E402
    profile_manifest_path,
    validate_manifest_profile,
)
from src.information_extraction.multitask_data import (  # noqa: E402
    CANONICAL_FIELD_LABELS,
    RELATION_LABELS,
)
from src.information_extraction.multitask_evaluation import (  # noqa: E402
    validate_evaluation_binding,
)
from src.ocr.environment import configure_external_environment, require_storage_gate  # noqa: E402
from src.rotation_common import (  # noqa: E402
    atomic_write_json,
    configuration_hash,
    read_csv_rows,
)
from scripts.train_multitask_model import (  # noqa: E402
    TokenizedWindowDataset,
    _device,
    _evaluate,
    _seed_everything,
    _sha256,
)

PUBLIC_EVALUATION_SPLITS = (
    "dev_select",
    "dev_calibration",
    "test_in_domain",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--profile", choices=("development", "final"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=PUBLIC_EVALUATION_SPLITS, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--rotation-angle",
        type=float,
        default=0.0,
        help="Rotate text/entity geometry together for a layout-robustness evaluation.",
    )
    parser.add_argument(
        "--streams",
        nargs="+",
        choices=("ground_truth", "paddleocr", "hybrid"),
        required=True,
    )
    parser.add_argument("--allow-cross-build", action="store_true")
    parser.add_argument(
        "--calibration",
        help="optional public calibration JSON; must be bound to this checkpoint/build",
    )
    parser.add_argument(
        "--group-by", nargs="*", choices=("dataset", "language"), default=(),
        help="also compute disjoint aggregate metrics for the requested public attributes",
    )
    parser.add_argument("--report-name", default=None)
    args = parser.parse_args()
    if args.max_length < 32:
        parser.error("--max-length must be at least 32")
    if args.report_name and not re.fullmatch(r"[A-Za-z0-9._-]+", args.report_name):
        parser.error("--report-name may contain only letters, digits, dot, underscore, and dash")

    cfg = cfgmod.load_config(args.config)
    asset_root = cfgmod.resolve_path(cfg, "external_assets")
    configure_external_environment(asset_root)
    require_storage_gate(
        asset_root,
        operation=f"{args.profile} public evaluation",
        anticipated_c_gib=0.25,
        anticipated_asset_gib=8.0,
    )

    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import LayoutXLMTokenizerFast

    from src.information_extraction.layoutxlm_model import MultiTaskTextLayoutModel

    seed = int(cfg.get("layout_model", {}).get("seed", 42))
    _seed_everything(seed, torch)
    selected_device = _device(args.device, torch)
    checkpoint = Path(args.checkpoint).resolve()
    state_path = checkpoint / "training_state.json"
    model_path = checkpoint / "model.safetensors"
    if not state_path.is_file() or not model_path.is_file():
        raise SystemExit(f"checkpoint is incomplete: {checkpoint}")
    training_state = json.loads(state_path.read_text(encoding="utf-8"))

    manifest_path = profile_manifest_path(
        cfgmod.resolve_path(cfg, "metadata"), args.profile
    )
    manifest_rows = read_csv_rows(manifest_path)
    build_ids = {row.get("build_id", "") for row in manifest_rows}
    if len(build_ids) != 1 or "" in build_ids:
        raise SystemExit(f"manifest build IDs are missing or mixed: {sorted(build_ids)!r}")
    evaluation_build_id = next(iter(build_ids))
    validate_manifest_profile(
        manifest_rows,
        expected_profile=args.profile,
        expected_build_id=evaluation_build_id,
    )
    try:
        validate_evaluation_binding(
            training_state,
            expected_profile=args.profile,
            evaluation_build_id=evaluation_build_id,
            allow_cross_build=bool(args.allow_cross_build),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    calibration: dict[str, Any] | None = None
    calibration_path: Path | None = None
    if args.calibration:
        calibration_path = Path(args.calibration)
        if not calibration_path.is_absolute():
            calibration_path = (PROJECT_ROOT / calibration_path).resolve()
        if not calibration_path.is_file():
            raise SystemExit(f"calibration report is missing: {calibration_path}")
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        if (
            calibration.get("profile") != args.profile
            or calibration.get("checkpoint_build_id") != evaluation_build_id
            or calibration.get("checkpoint_model_sha256") != _sha256(model_path)
            or calibration.get("public_only") is not True
            or calibration.get("private_example_count") != 0
        ):
            raise SystemExit("calibration is not bound to this public checkpoint/build")

    token_sources = set(args.streams)
    examples = load_model_examples(
        manifest_path,
        args.split,
        expected_profile=args.profile,
        expected_build_id=evaluation_build_id,
        token_sources=token_sources,
    )
    if not examples:
        raise SystemExit("the requested public split/stream selection has no examples")
    if any(example.get("is_private") is not False for example in examples):
        raise SystemExit("private or unmarked evaluation example detected")

    tokenizer = LayoutXLMTokenizerFast.from_pretrained(checkpoint)
    model = MultiTaskTextLayoutModel.from_pretrained(checkpoint).to(selected_device).eval()
    cache_signature = configuration_hash({
        "schema": "multitask-window-v1",
        "build_id": evaluation_build_id,
        "max_length": args.max_length,
        "stride": 64,
        "tokenizer": "microsoft/layoutxlm-base",
        "entity_labels": BIO_LABELS,
        "canonical_labels": CANONICAL_FIELD_LABELS,
        "relation_labels": RELATION_LABELS,
        "fixed_rotation_angle": args.rotation_angle,
    })
    dataset = TokenizedWindowDataset(
        examples,
        tokenizer,
        cfgmod.resolve_path(cfg, "model_datasets")
        / args.profile
        / evaluation_build_id
        / "tokenized"
        / cache_signature
        / args.split,
        max_length=args.max_length,
        seed=seed,
        torch_module=torch,
        dataset_base=Dataset,
        fixed_rotation_angle=args.rotation_angle,
    )
    metrics = _evaluate(
        model, dataset, selected_device, torch, DataLoader, calibration=calibration
    )
    grouped_metrics: dict[str, dict[str, Any]] = {}
    for field in args.group_by:
        grouped_metrics[field] = {}
        values = sorted({str(example.get(field, "unknown")) for example in examples})
        for value in values:
            group_examples = [
                example for example in examples if str(example.get(field, "unknown")) == value
            ]
            safe_value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
            group_dataset = TokenizedWindowDataset(
                group_examples,
                tokenizer,
                cfgmod.resolve_path(cfg, "model_datasets")
                / args.profile
                / evaluation_build_id
                / "tokenized"
                / cache_signature
                / args.split
                / "groups"
                / field
                / safe_value,
                max_length=args.max_length,
                seed=seed,
                torch_module=torch,
                dataset_base=Dataset,
                fixed_rotation_angle=args.rotation_angle,
            )
            grouped_metrics[field][value] = {
                "example_count": len(group_examples),
                "window_count": len(group_dataset),
                "metrics": _evaluate(
                    model, group_dataset, selected_device, torch, DataLoader,
                    calibration=calibration,
                ),
            }
    report = {
        "schema_version": "1.0",
        "profile": args.profile,
        "split": args.split,
        "token_sources": sorted(token_sources),
        "public_only": True,
        "private_example_count": 0,
        "checkpoint": str(checkpoint),
        "checkpoint_model_sha256": _sha256(model_path),
        "checkpoint_build_id": str(training_state.get("build_id", "")),
        "evaluation_build_id": evaluation_build_id,
        "cross_build_comparison": str(training_state.get("build_id", "")) != evaluation_build_id,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "example_count": len(examples),
        "window_count": len(dataset),
        "rotation_angle": float(args.rotation_angle) % 360.0,
        "device": str(selected_device),
        "calibration": (
            {
                "path": str(calibration_path),
                "sha256": _sha256(calibration_path),
                "split": calibration.get("split"),
            }
            if calibration is not None and calibration_path is not None
            else None
        ),
        "metrics": metrics,
        "grouped_metrics": grouped_metrics,
    }
    report_name = args.report_name or (
        f"{checkpoint.name}_{args.split}_{'-'.join(sorted(token_sources))}.json"
    )
    report_path = cfgmod.resolve_path(cfg, "reports") / "final_model" / "evaluations" / report_name
    atomic_write_json(report_path, report)
    print(json.dumps({**report, "report_path": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
