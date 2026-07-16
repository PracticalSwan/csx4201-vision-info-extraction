#!/usr/bin/env python3
"""Evaluate final layout-head quality across the required fixed-angle grid."""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config as cfgmod  # noqa: E402
from src.information_extraction.layoutxlm_data import BIO_LABELS, load_model_examples  # noqa: E402
from src.information_extraction.model_dataset import profile_manifest_path, validate_manifest_profile  # noqa: E402
from src.information_extraction.multitask_data import CANONICAL_FIELD_LABELS, RELATION_LABELS  # noqa: E402
from src.information_extraction.multitask_evaluation import validate_evaluation_binding  # noqa: E402
from src.ocr.environment import configure_external_environment, require_storage_gate  # noqa: E402
from src.rotation_common import atomic_write_json, configuration_hash, deterministic_rank, read_csv_rows, sha256_file  # noqa: E402
from scripts.train_multitask_model import TokenizedWindowDataset, _device, _evaluate, _seed_everything  # noqa: E402

ANGLES = (0, 1, 15, 30, 37, 45, 60, 89, 90, 91, 135, 179, 180, 225, 269, 270, 315, 359)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--calibration", default=str(PROJECT_ROOT / "models" / "multitask_calibration.json")
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--pages-per-dataset", type=int, default=10)
    args = parser.parse_args()
    if args.pages_per_dataset < 1:
        parser.error("--pages-per-dataset must be positive")
    cfg = cfgmod.load_config(args.config)
    asset_root = cfgmod.resolve_path(cfg, "external_assets")
    configure_external_environment(asset_root)
    require_storage_gate(
        asset_root, operation="fixed-angle final layout evaluation",
        anticipated_c_gib=0.25, anticipated_asset_gib=12.0,
    )

    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import LayoutXLMTokenizerFast
    from src.information_extraction.layoutxlm_model import MultiTaskTextLayoutModel

    checkpoint = Path(args.checkpoint).resolve()
    state = json.loads((checkpoint / "training_state.json").read_text(encoding="utf-8"))
    calibration_path = Path(args.calibration).resolve()
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    manifest_path = profile_manifest_path(cfgmod.resolve_path(cfg, "metadata"), "final")
    rows = read_csv_rows(manifest_path)
    build_ids = {row.get("build_id", "") for row in rows}
    if len(build_ids) != 1 or "" in build_ids:
        raise SystemExit("final manifest has missing or mixed build IDs")
    build_id = next(iter(build_ids))
    validate_manifest_profile(rows, expected_profile="final", expected_build_id=build_id)
    validate_evaluation_binding(
        state, expected_profile="final", evaluation_build_id=build_id, allow_cross_build=False
    )
    if (
        calibration.get("profile") != "final"
        or calibration.get("checkpoint_build_id") != build_id
        or calibration.get("checkpoint_model_sha256")
        != sha256_file(checkpoint / "model.safetensors")
        or calibration.get("private_example_count") != 0
    ):
        raise SystemExit("calibration is not bound to the final public checkpoint/build")
    examples = load_model_examples(
        manifest_path, "test_in_domain", expected_profile="final",
        expected_build_id=build_id, token_sources={"ground_truth"},
    )
    examples = _balanced_examples(examples, args.pages_per_dataset)
    if any(example.get("is_private") is not False for example in examples):
        raise SystemExit("private or unmarked angle-evaluation example detected")

    seed = int(cfg.get("layout_model", {}).get("seed", 42))
    _seed_everything(seed, torch)
    device = _device(args.device, torch)
    tokenizer = LayoutXLMTokenizerFast.from_pretrained(checkpoint)
    model = MultiTaskTextLayoutModel.from_pretrained(checkpoint).to(device).eval()
    started = time.perf_counter()
    observations = []
    for angle in ANGLES:
        signature = configuration_hash({
            "schema": "final-angle-evaluation-v1", "build_id": build_id,
            "angle": angle, "example_ids": sorted(example["example_id"] for example in examples),
            "entity_labels": BIO_LABELS, "canonical_labels": CANONICAL_FIELD_LABELS,
            "relation_labels": RELATION_LABELS,
        })
        dataset = TokenizedWindowDataset(
            examples, tokenizer,
            cfgmod.resolve_path(cfg, "model_datasets") / "final" / build_id
            / "tokenized" / signature / f"angle_{angle}",
            max_length=512, seed=seed, torch_module=torch, dataset_base=Dataset,
            fixed_rotation_angle=float(angle),
        )
        observations.append({
            "angle": angle,
            "example_count": len(examples),
            "window_count": len(dataset),
            "metrics": _evaluate(
                model, dataset, device, torch, DataLoader, calibration=calibration
            ),
        })
    baseline = observations[0]["metrics"]
    for item in observations:
        metrics = item["metrics"]
        item["retention_vs_upright"] = {
            name: (
                float(metrics[name]) / float(baseline[name])
                if float(baseline.get(name, 0.0) or 0.0) > 0 else None
            )
            for name in (
                "entity_token_f1", "canonical_evidence_token_f1", "relation_f1",
                "composite_score", "entity_calibrated_f1", "canonical_calibrated_f1",
                "relation_calibrated_f1", "calibrated_composite_score",
            )
        }
    report = {
        "schema_version": "1.0",
        "profile": "final",
        "split": "test_in_domain",
        "public_only": True,
        "private_example_count": 0,
        "checkpoint": str(checkpoint),
        "checkpoint_model_sha256": sha256_file(checkpoint / "model.safetensors"),
        "calibration_sha256": sha256_file(calibration_path),
        "build_id": build_id,
        "manifest_sha256": sha256_file(manifest_path),
        "sample_strategy": "deterministic dataset-balanced ground-truth geometry sample",
        "pages_per_dataset": args.pages_per_dataset,
        "example_count": len(examples),
        "datasets": sorted({example["dataset"] for example in examples}),
        "angles": observations,
        "duration_seconds": time.perf_counter() - started,
        "limitations": [
            "This isolates text-layout head robustness by rotating token and entity geometry together; end-to-end OCR rotation is evaluated separately."
        ],
    }
    output = cfgmod.resolve_path(cfg, "reports") / "final_model" / "layout_angle_metrics.json"
    atomic_write_json(output, report)
    print(json.dumps(report, indent=2))
    return 0


def _balanced_examples(examples: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        buckets[str(example.get("dataset", "unknown"))].append(example)
    selected = []
    for dataset in sorted(buckets):
        ordered = sorted(
            buckets[dataset],
            key=lambda example: deterministic_rank(str(example["example_id"]), 4242),
        )
        selected.extend(ordered[:limit])
    return selected


if __name__ == "__main__":
    raise SystemExit(main())
