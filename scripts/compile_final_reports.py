#!/usr/bin/env python3
"""Compile required final-model reports from executed, hash-bound artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config as cfgmod  # noqa: E402
from src.rotation_common import atomic_write_json, atomic_write_text, sha256_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument(
        "--heldout-report", default="final_test_in_domain_ground_truth.json"
    )
    args = parser.parse_args()
    cfg = cfgmod.load_config(args.config)
    root = cfgmod.resolve_path(cfg, "reports") / "final_model"
    heldout_path = root / "evaluations" / args.heldout_report
    required = {
        "dataset": root / "dataset_summary.json",
        "model_dataset": root / "model_dataset_final_summary.json",
        "training": root / "training_summary.json",
        "heldout": heldout_path,
        "calibration": PROJECT_ROOT / "models" / "multitask_calibration.json",
        "ocr_ablation": root / "ocr_preprocessing_ablation.json",
        "layout_angles": root / "layout_angle_metrics.json",
        "end_to_end_angles": root / "end_to_end_angle_metrics.json",
        "unseen": root / "unseen_domain_metrics.json",
        "private": root / "private_test_aggregate.json",
        "integration": PROJECT_ROOT / "reports" / "information_extraction" / "integration_smoke.json",
        "ocr_models": PROJECT_ROOT / "reports" / "ocr" / "model_verification.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"final report inputs are missing: {missing}")
    artifacts = {name: _read(path) for name, path in required.items()}
    heldout = artifacts["heldout"]
    metrics = heldout["metrics"]
    grouped = heldout.get("grouped_metrics") or {}
    end_angles = artifacts["end_to_end_angles"]
    upright = next(item for item in end_angles["public_metrics"] if item["angle"] == 0)

    ocr_metrics = {
        "schema_version": "1.0",
        "heldout_scope": "bounded end-to-end public test sample; development ablation reported separately",
        "upright": {
            key: upright.get(key)
            for key in ("page_count", "recognized_text_coverage", "wer", "detection_f1", "nonempty_rate")
        },
        "preprocessing_ablation": artifacts["ocr_ablation"],
        "exact_model_verification": artifacts["ocr_models"],
        "model_hashes": artifacts["model_dataset"].get("build_provenance", {}).get("ocr_model_artifact_hashes"),
    }
    atomic_write_json(root / "ocr_metrics.json", ocr_metrics)

    entity_metrics = _head_report(
        heldout, "entity_token_f1", "entity_macro_f1", "entity_per_class", "entity_counts",
        "entity_calibrated_f1", "entity_calibrated_macro_f1",
        "entity_calibrated_per_class", "entity_calibrated_counts",
    )
    relation_metrics = _head_report(
        heldout, "relation_f1", "relation_macro_f1", "relation_per_class", "relation_counts",
        "relation_calibrated_f1", "relation_calibrated_macro_f1",
        "relation_calibrated_per_class", "relation_calibrated_counts",
    )
    atomic_write_json(root / "entity_metrics.json", entity_metrics)
    atomic_write_json(root / "relation_metrics.json", relation_metrics)

    field_metrics = {
        "schema_version": "1.0",
        "split": "test_in_domain",
        "canonical_evidence_token_f1": metrics.get("canonical_evidence_token_f1"),
        "canonical_macro_f1": metrics.get("canonical_macro_f1"),
        "canonical_counts": metrics.get("canonical_counts"),
        "canonical_per_class": metrics.get("canonical_per_class"),
        "canonical_calibrated_f1": metrics.get("canonical_calibrated_f1"),
        "canonical_calibrated_macro_f1": metrics.get("canonical_calibrated_macro_f1"),
        "canonical_calibrated_counts": metrics.get("canonical_calibrated_counts"),
        "canonical_calibrated_per_class": metrics.get("canonical_calibrated_per_class"),
        "bounded_end_to_end_upright_field_accuracy": upright.get("field_accuracy"),
        "unseen_coru_canonical_field_accuracy": artifacts["unseen"].get("canonical_field_accuracy"),
        "calibration": {
            "thresholds": artifacts["calibration"].get("thresholds"),
            "canonical_field_thresholds": artifacts["calibration"].get("canonical_field_thresholds"),
            "metrics": artifacts["calibration"].get("metrics"),
        },
    }
    atomic_write_json(root / "field_metrics.json", field_metrics)

    angle_metrics = {
        "schema_version": "1.0",
        "layout_head_geometry": artifacts["layout_angles"],
        "end_to_end_ocr_and_extraction": end_angles,
        "kmeans_controls_ocr": False,
    }
    atomic_write_json(root / "angle_metrics.json", angle_metrics)

    language_metrics = {
        "schema_version": "1.0",
        "heldout_layout_by_language": grouped.get("language", {}),
        "end_to_end_route_counts_by_angle": {
            str(item["angle"]): item["route_counts"] for item in end_angles["public_metrics"]
        },
        "synthetic_thai": end_angles["synthetic_thai_metrics"],
        "thai_limitation": "Synthetic Thai is an operational route/rotation check, not a labeled public Thai accuracy benchmark.",
    }
    atomic_write_json(root / "language_metrics.json", language_metrics)

    dataset_metrics = {
        "schema_version": "1.0",
        "heldout_layout_by_dataset": grouped.get("dataset", {}),
        "bounded_end_to_end_by_angle_and_dataset": {
            str(item["angle"]): item.get("by_dataset", {})
            for item in end_angles["public_metrics"]
        },
        "unseen_coru": artifacts["unseen"],
        "dataset_inventory": artifacts["dataset"]["datasets"],
    }
    atomic_write_json(root / "dataset_metrics.json", dataset_metrics)
    atomic_write_text(root / "error_analysis.md", _error_analysis(artifacts, metrics))
    atomic_write_text(root / "final_model_card.md", _model_card(artifacts, metrics, upright))

    verification = {
        "schema_version": "1.0",
        "status": "compiled_from_executed_artifacts",
        "checkpoint_reload_passed": artifacts["training"].get("checkpoint_reload_passed"),
        "final_profile_trained": artifacts["training"].get("profile") == "final",
        "usable_public_fit_pages": artifacts["dataset"].get("usable_public_fit_pages"),
        "gmail_fit_rows": artifacts["dataset"].get("gmail_fit_rows"),
        "private_operational_test_successes": artifacts["private"].get("successful_documents"),
        "integration_status": artifacts["integration"].get("status"),
        "kmeans_controls_ocr": False,
        "artifact_hashes": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in required.items()
        },
        "limitations": [
            "Passing lifecycle verification does not imply production quality; measured held-out metrics and model-card limitations remain authoritative."
        ],
    }
    atomic_write_json(root / "verification.json", verification)
    print(json.dumps(verification, indent=2))
    return 0


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _head_report(
    heldout: dict[str, Any], micro_key: str, macro_key: str,
    classes_key: str, counts_key: str, calibrated_micro_key: str,
    calibrated_macro_key: str, calibrated_classes_key: str,
    calibrated_counts_key: str,
) -> dict[str, Any]:
    metrics = heldout["metrics"]
    return {
        "schema_version": "1.0",
        "profile": heldout["profile"],
        "split": heldout["split"],
        "example_count": heldout["example_count"],
        "micro_f1": metrics.get(micro_key),
        "macro_f1": metrics.get(macro_key),
        "counts": metrics.get(counts_key),
        "per_class": metrics.get(classes_key),
        "calibrated_micro_f1": metrics.get(calibrated_micro_key),
        "calibrated_macro_f1": metrics.get(calibrated_macro_key),
        "calibrated_counts": metrics.get(calibrated_counts_key),
        "calibrated_per_class": metrics.get(calibrated_classes_key),
        "by_dataset": heldout.get("grouped_metrics", {}).get("dataset", {}),
        "by_language": heldout.get("grouped_metrics", {}).get("language", {}),
    }


def _error_analysis(artifacts: dict[str, dict[str, Any]], metrics: dict[str, Any]) -> str:
    end = artifacts["end_to_end_angles"]
    totals: dict[str, int] = {}
    for item in end["public_metrics"]:
        for name, value in item.get("error_counts", {}).items():
            totals[name] = totals.get(name, 0) + int(value)
    ranked = sorted(totals.items(), key=lambda item: (-item[1], item[0]))
    weakest_entity = _weakest_classes(metrics.get("entity_per_class") or {})
    weakest_relation = _weakest_classes(metrics.get("relation_per_class") or {})
    weakest_canonical = _weakest_classes(metrics.get("canonical_per_class") or {})
    dataset_rows = []
    for dataset, values in sorted(
        artifacts["heldout"].get("grouped_metrics", {}).get("dataset", {}).items()
    ):
        item = values.get("metrics") or {}
        dataset_rows.append(
            f"| {dataset} | {values.get('example_count')} | "
            f"{item.get('entity_token_f1')!r} | {item.get('relation_f1')!r} | "
            f"{item.get('canonical_evidence_token_f1')!r} |"
        )
    weakest_angles = sorted(
        end["public_metrics"],
        key=lambda item: (item.get("recognized_text_coverage", 0.0), item["angle"]),
    )[:5]
    lines = [
        "# Final public error analysis",
        "",
        "This analysis uses executed public test predictions. Counts below come from the bounded end-to-end angle grid; no private text or filenames are included.",
        "",
        "| Root-cause signal | Count |",
        "|---|---:|",
        *[f"| {name.replace('_', ' ')} | {count} |" for name, count in ranked],
        "",
        "## Weakest supported classes on the full locked layout test",
        "",
        "| Head | Class | Support | F1 |",
        "|---|---|---:|---:|",
        *[f"| entity | {name} | {support} | {score:.4f} |" for name, support, score in weakest_entity],
        *[f"| relation | {name} | {support} | {score:.4f} |" for name, support, score in weakest_relation],
        *[f"| canonical evidence | {name} | {support} | {score:.4f} |" for name, support, score in weakest_canonical],
        "",
        "## Locked-test dataset slices",
        "",
        "| Dataset | Examples | Entity F1 | Relation F1 | Canonical evidence F1 |",
        "|---|---:|---:|---:|---:|",
        *dataset_rows,
        "",
        "## Lowest bounded OCR coverage angles",
        "",
        "| Angle | Pages | Recognized-text coverage | WER | Entity F1 |",
        "|---:|---:|---:|---:|---:|",
        *[
            f"| {item['angle']} | {item['page_count']} | "
            f"{item.get('recognized_text_coverage')!r} | {item.get('wer')!r} | "
            f"{item.get('entity_f1')!r} |"
            for item in weakest_angles
        ],
        "",
        "These are bounded diagnostic signals, not all ground-truth error labels: for example, `no table detected` records output availability because the sampled annotations do not provide a compatible table benchmark. Implemented mitigations include cardinal-plus-polygon fine deskew, real PaddleOCR/hybrid training streams, class-weighted multi-task loss, calibrated abstention, arithmetic validation, and geometry table fallback. Remaining misses stay visible in the measured metrics.",
        "",
        f"Held-out entity micro-F1: {metrics.get('entity_token_f1')!r}.",
        f"Held-out calibrated entity micro-F1: {metrics.get('entity_calibrated_f1')!r}.",
        f"Held-out relation F1: {metrics.get('relation_f1')!r}.",
        f"Held-out calibrated relation F1: {metrics.get('relation_calibrated_f1')!r}.",
        f"Held-out canonical evidence F1: {metrics.get('canonical_evidence_token_f1')!r}.",
        f"Held-out calibrated canonical evidence F1: {metrics.get('canonical_calibrated_f1')!r}.",
        "",
        "Known bottlenecks: relation labels exist only in FUNSD; the Windows runtime has no compatible Detectron2 visual backbone; CORU QA has no token polygons; and the public corpus has no compatible labeled Thai benchmark. These constraints are not treated as successes.",
    ]
    return "\n".join(lines) + "\n"


def _weakest_classes(values: dict[str, Any], limit: int = 5) -> list[tuple[str, int, float]]:
    supported = [
        (name, int(metrics.get("support", 0)), float(metrics.get("f1", 0.0)))
        for name, metrics in values.items()
        if int(metrics.get("support", 0)) > 0
    ]
    return sorted(supported, key=lambda item: (item[2], -item[1], item[0]))[:limit]


def _model_card(
    artifacts: dict[str, dict[str, Any]], metrics: dict[str, Any], upright: dict[str, Any]
) -> str:
    training = artifacts["training"]
    dataset = artifacts["dataset"]
    unseen = artifacts["unseen"]
    checkpoint = Path(str(training.get("checkpoint", "")))
    model_path = checkpoint / "model.safetensors"
    checkpoint_sha = sha256_file(model_path) if model_path.is_file() else "unavailable"
    return "\n".join([
        "# Final vision information-extraction pre-model card",
        "",
        "## Model",
        "",
        "The checkpoint is a LayoutXLM-initialized multilingual text-plus-normalized-2D-layout encoder with trained entity, document-type, canonical-evidence, and real relation heads. PaddleOCR runs in an isolated process path with exact general and Thai recognizers. The K-Means model is display-only.",
        "",
        f"Checkpoint: `{training.get('checkpoint')}`.",
        f"Checkpoint model SHA-256: `{checkpoint_sha}`.",
        f"Training: {training.get('completed_epochs')} completed epochs, {training.get('optimizer_steps')} optimizer steps; best epoch {training.get('best_epoch')} with selection score {training.get('best_composite_score')!r}.",
        f"License inherited from the base checkpoint: `{training.get('license', 'CC-BY-NC-SA-4.0')}`.",
        "",
        "## Training data and privacy",
        "",
        f"Public fit pages: {dataset.get('usable_public_fit_pages')}; examples: {dataset.get('usable_examples')}; Gmail/private fit rows: {dataset.get('gmail_fit_rows')}.",
        "Private Gmail documents are operational test only and never train, calibrate, or select the model.",
        "",
        "## Measured quality",
        "",
        f"Held-out entity micro-F1: raw {metrics.get('entity_token_f1')!r}; calibrated/abstained {metrics.get('entity_calibrated_f1')!r}; raw macro-F1 {metrics.get('entity_macro_f1')!r}.",
        f"Held-out relation F1: raw {metrics.get('relation_f1')!r}; calibrated/abstained {metrics.get('relation_calibrated_f1')!r}.",
        f"Held-out canonical evidence F1: raw {metrics.get('canonical_evidence_token_f1')!r}; calibrated/abstained {metrics.get('canonical_calibrated_f1')!r}.",
        f"Bounded upright end-to-end OCR text coverage: {upright.get('recognized_text_coverage')!r}; WER: {upright.get('wer')!r}.",
        f"CORU unseen-domain answer-text recall: {unseen.get('qa_answer_text_recall')!r} on {unseen.get('sample_pages')} sampled pages.",
        "",
        "## Intended use",
        "",
        "Local extraction from images and PDFs containing receipts, invoices, forms, and unfamiliar documents. Outputs include OCR evidence, generic entities and relations, canonical fields with abstention, and geometry-based tables.",
        "",
        "## Limitations",
        "",
        "This is a bounded academic pre-model, not a production or high-stakes decision system. The visual backbone is unavailable on this Windows runtime, relation supervision is sparse, Thai quality lacks a labeled public benchmark, arbitrary-angle end-to-end evaluation is bounded, and low-confidence/unsupported fields return null. Review financial and legal outputs against the source document.",
    ]) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
