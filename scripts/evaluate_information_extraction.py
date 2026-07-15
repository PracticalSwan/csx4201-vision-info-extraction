#!/usr/bin/env python3
"""Run bounded public upright/rotated OCR and extraction evaluation."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image  # noqa: E402

from src import config as cfgmod  # noqa: E402
from src.evaluation.metrics import extraction_metrics, ocr_text_metrics, text_detection_metrics  # noqa: E402
from src.information_extraction.geometry import (  # noqa: E402
    expanded_rotation_transform,
    rotate_image,
    transform_annotation,
)
from src.inference.document_io import DocumentPage  # noqa: E402
from src.inference.document_pipeline import DocumentPipeline  # noqa: E402
from src.ocr.environment import configure_external_environment  # noqa: E402
from src.rotation_common import atomic_write_csv, atomic_write_json, read_csv_rows  # noqa: E402

SMOKE_ANGLES = [0.0, 45.0, 90.0, 270.0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--angles", help="comma-separated counterclockwise input rotations")
    parser.add_argument("--pages-per-dataset", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "gpu:0"), default="gpu:0")
    parser.add_argument("--layout-checkpoint")
    parser.add_argument("--model-setup", default=str(PROJECT_ROOT / "reports" / "ocr" / "model_setup.json"))
    args = parser.parse_args()
    if args.pages_per_dataset < 1:
        parser.error("--pages-per-dataset must be positive")
    cfg = cfgmod.load_config(args.config)
    configure_external_environment(cfgmod.resolve_path(cfg, "external_assets"))
    angles = _angles(args.angles) if args.angles else (
        SMOKE_ANGLES if args.profile == "smoke" else [float(value) for value in cfg["augmentation"]["validation_angles"]]
    )
    manifest = read_csv_rows(cfgmod.resolve_path(cfg, "metadata") / "information_extraction_manifest.csv")
    model_manifest = read_csv_rows(cfgmod.resolve_path(cfg, "metadata") / "model_dataset_manifest.csv")
    fit_datasets = sorted({
        row["dataset"] for row in model_manifest
        if row.get("is_usable") == "true" and row.get("project_split") == "train"
    })
    gmail_fit_rows = sum(
        row.get("is_private") == "true" and row.get("project_split") == "train"
        for row in model_manifest
    )
    selected = _select_examples(manifest, args.pages_per_dataset)
    pipeline = DocumentPipeline.from_config(
        cfg,
        device=args.device,
        model_setup=args.model_setup,
        layout_checkpoint=args.layout_checkpoint,
        enable_kmeans_display=True,
    )
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for source in selected:
            annotation = json.loads((PROJECT_ROOT / source["normalized_annotation_path"]).read_text(encoding="utf-8"))
            reference_text = " ".join(str(token["text"]) for token in annotation.get("tokens") or [])
            with Image.open(PROJECT_ROOT / source["image_path"]) as raw:
                original = raw.convert("RGB")
            for angle in angles:
                transform = expanded_rotation_transform(original.width, original.height, angle)
                rotated = rotate_image(original, transform)
                rotated_annotation = transform_annotation(annotation, transform)
                try:
                    result = pipeline.extract_pages(
                        document_id=f"eval_{source['page_id']}_{angle:g}",
                        source_type="image",
                        pages=[DocumentPage(1, rotated)],
                        language="auto",
                        language_hint=source.get("language"),
                    )
                    page = result["pages"][0]
                    ocr_metrics = ocr_text_metrics(reference_text, page["full_text"])
                    ocr_reference_available = bool(ocr_metrics["reference_characters"])
                    detection = text_detection_metrics(
                        rotated_annotation.get("tokens") or [], page["ocr"]["words"]
                    )
                    ie_metrics = extraction_metrics(annotation, page, result["fields"])
                    expected_correction = (-angle) % 360.0
                    selected_orientation = float(page["selected_ocr_orientation"])
                    orientation_error = _circular_error(selected_orientation, expected_correction)
                    rows.append({
                        "dataset": source["dataset"],
                        "language": source.get("language", "unknown"),
                        "page_id": source["page_id"],
                        "input_angle": angle,
                        "selected_orientation": selected_orientation,
                        "orientation_error": orientation_error,
                        "orientation_exact": orientation_error < 1e-6,
                        "cer": ocr_metrics["cer"] if ocr_reference_available else None,
                        "wer": ocr_metrics["wer"] if ocr_reference_available else None,
                        "ocr_reference_available": ocr_reference_available,
                        "recognized_text_coverage": (
                            ocr_metrics["recognized_text_coverage"]
                            if ocr_reference_available else None
                        ),
                        "detection_reference_available": detection["reference_available"],
                        "detection_tp": detection["true_positive"],
                        "detection_expected": detection["expected"],
                        "detection_predicted": detection["predicted"],
                        "character_errors": ocr_metrics["character_errors"],
                        "reference_characters": ocr_metrics["reference_characters"],
                        "word_errors": ocr_metrics["word_errors"],
                        "reference_words": len(reference_text.split()),
                        "empty_output": ocr_metrics["empty_output"],
                        "ocr_word_count": len(page["ocr"]["words"]),
                        "mean_ocr_confidence": page["ocr"]["mean_confidence"],
                        "duration_seconds": result["processing"]["duration_seconds"],
                        "entity_tp": ie_metrics["entity"]["true_positive"],
                        "entity_expected": ie_metrics["entity"]["expected"],
                        "entity_predicted": ie_metrics["entity"]["predicted"],
                        "relation_tp": ie_metrics["relation"]["true_positive"],
                        "relation_expected": ie_metrics["relation"]["expected"],
                        "relation_predicted": ie_metrics["relation"]["predicted"],
                        "key_value_pair_count": len(page["key_value_pairs"]),
                        "field_correct": ie_metrics["canonical_fields"]["correct"],
                        "field_applicable": ie_metrics["canonical_fields"]["applicable"],
                        "document_type_correct": int(
                            str(result["document_type"]["label"]).casefold()
                            == str(source.get("document_type", "unknown")).casefold()
                        ),
                        "kmeans_zone": result["rotation_display"]["zone"],
                        "kmeans_confidence": result["rotation_display"]["confidence"],
                    })
                except Exception as exc:
                    errors.append({
                        "dataset": source["dataset"], "page_id": source["page_id"],
                        "input_angle": angle, "error": f"{type(exc).__name__}: {exc}",
                    })
    finally:
        pipeline.close()

    aggregate = _aggregate(rows)
    by_angle = {str(angle): _aggregate([row for row in rows if row["input_angle"] == angle]) for angle in angles}
    by_dataset = {
        dataset: _aggregate([row for row in rows if row["dataset"] == dataset])
        for dataset in sorted({row["dataset"] for row in rows})
    }
    by_language = {
        language: _aggregate([row for row in rows if row["language"] == language])
        for language in sorted({row["language"] for row in rows})
    }
    upright = _aggregate([row for row in rows if row["input_angle"] == 0.0])
    rotated = _aggregate([row for row in rows if row["input_angle"] != 0.0])
    unseen_rows = [row for row in rows if row["dataset"] not in fit_datasets]
    unseen_datasets = sorted({row["dataset"] for row in unseen_rows})
    unseen_protocol = {
        "status": "executed_natural_dataset_holdout" if unseen_rows else "not_available",
        "datasets": unseen_datasets,
        "training_datasets": fit_datasets,
        "fit_rows_for_evaluated_datasets": 0,
        "sample_count": len(unseen_rows),
        "generic_ocr_nonempty_rate": (
            sum(not row["empty_output"] for row in unseen_rows) / len(unseen_rows)
            if unseen_rows else None
        ),
        "mean_ocr_words": _mean([row["ocr_word_count"] for row in unseen_rows]),
        "mean_predicted_entities": _mean([row["entity_predicted"] for row in unseen_rows]),
        "mean_key_value_pairs": _mean([row["key_value_pair_count"] for row in unseen_rows]),
        "metrics": _aggregate(unseen_rows),
        "scope": (
            "The smoke checkpoint had no usable training rows from these datasets; this is a "
            "natural public holdout, not a separately tuned leave-one-dataset-out run."
        ),
    }
    upright_f1 = upright.get("entity_f1", 0.0)
    rotation_retention = (
        rotated.get("entity_f1", 0.0) / upright_f1 if upright_f1 else None
    )
    report = {
        "schema_version": "1.0",
        "status": f"{args.profile}_evaluation",
        "profile": args.profile,
        "public_only": True,
        "gmail_fit_rows": gmail_fit_rows,
        "selected_public_pages": len(selected),
        "datasets_selected": sorted({row["dataset"] for row in selected}),
        "angles": angles,
        "successful_runs": len(rows),
        "failed_runs": len(errors),
        "aggregate": aggregate,
        "upright": upright,
        "rotated": rotated,
        "rotation_retention": rotation_retention,
        "by_angle": by_angle,
        "by_dataset": by_dataset,
        "by_language": by_language,
        "unseen_document_protocol": unseen_protocol,
        "errors": errors,
        "duration_seconds": time.perf_counter() - started,
        "limitations": [
            "This bounded evaluation is not a final-quality benchmark." if args.profile == "smoke" else "Full angle coverage does not substitute for per-dataset holdout retraining.",
            "K-Means values are reported separately and never control OCR candidate selection.",
        ],
    }
    report_root = cfgmod.resolve_path(cfg, "reports")
    atomic_write_json(report_root / "model_evaluation" / f"{args.profile}_evaluation.json", report)
    atomic_write_json(report_root / "ocr" / f"{args.profile}_ocr_evaluation.json", {
        "profile": args.profile, "aggregate": aggregate, "by_angle": by_angle,
        "by_dataset": by_dataset, "by_language": by_language, "errors": errors,
    })
    atomic_write_json(report_root / "information_extraction" / f"{args.profile}_ie_evaluation.json", {
        "profile": args.profile, "aggregate": aggregate, "upright": upright,
        "rotated": rotated, "rotation_retention": rotation_retention,
        "by_angle": by_angle, "by_dataset": by_dataset, "by_language": by_language,
    })
    columns = list(rows[0]) if rows else ["dataset", "page_id", "input_angle"]
    atomic_write_csv(report_root / "model_evaluation" / f"{args.profile}_predictions.csv", rows, columns)
    print(json.dumps(report, indent=2))
    return 0 if rows and not errors else 1


def _select_examples(rows: list[dict[str, str]], per_dataset: int) -> list[dict[str, str]]:
    buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if (
            row.get("is_private") == "false"
            and row.get("is_usable") == "true"
            and row.get("normalized_annotation_path")
            and row.get("image_path")
        ):
            buckets[row["dataset"]].append(row)
    selected = []
    for dataset in sorted(buckets):
        ranked = sorted(
            buckets[dataset],
            key=lambda row: (
                row.get("project_split") != "test",
                not row.get("has_field_annotation") == "true",
                row["page_id"],
            ),
        )
        selected.extend(ranked[:per_dataset])
    return selected


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"sample_count": 0}
    entity_tp = sum(int(row["entity_tp"]) for row in rows)
    entity_expected = sum(int(row["entity_expected"]) for row in rows)
    entity_predicted = sum(int(row["entity_predicted"]) for row in rows)
    relation_tp = sum(int(row["relation_tp"]) for row in rows)
    relation_expected = sum(int(row["relation_expected"]) for row in rows)
    relation_predicted = sum(int(row["relation_predicted"]) for row in rows)
    ocr_reference_rows = [row for row in rows if row["ocr_reference_available"]]
    detection_reference_rows = [row for row in rows if row["detection_reference_available"]]
    reference_characters = sum(row["reference_characters"] for row in ocr_reference_rows)
    reference_words = sum(row["reference_words"] for row in ocr_reference_rows)
    detection_tp = sum(row["detection_tp"] for row in detection_reference_rows)
    detection_expected = sum(row["detection_expected"] for row in detection_reference_rows)
    detection_predicted = sum(row["detection_predicted"] for row in detection_reference_rows)
    detection_precision = detection_tp / detection_predicted if detection_predicted else 0.0
    detection_recall = detection_tp / detection_expected if detection_expected else None
    return {
        "sample_count": len(rows),
        "ocr_reference_sample_count": len(ocr_reference_rows),
        "ocr_reference_coverage": len(ocr_reference_rows) / len(rows),
        "cer": (
            sum(row["character_errors"] for row in ocr_reference_rows) / reference_characters
            if reference_characters else None
        ),
        "wer": (
            sum(row["word_errors"] for row in ocr_reference_rows) / reference_words
            if reference_words else None
        ),
        "recognized_text_coverage": (
            max(0.0, 1.0 - sum(row["character_errors"] for row in ocr_reference_rows) / reference_characters)
            if reference_characters else None
        ),
        "text_detection_reference_sample_count": len(detection_reference_rows),
        "text_detection_reference_coverage": len(detection_reference_rows) / len(rows),
        "text_detection_precision": detection_precision if detection_expected else None,
        "text_detection_recall": detection_recall,
        "text_detection_f1": (
            2.0 * detection_precision * detection_recall / (detection_precision + detection_recall)
            if detection_recall is not None and detection_precision + detection_recall
            else 0.0 if detection_recall is not None else None
        ),
        "empty_output_rate": sum(row["empty_output"] for row in rows) / len(rows),
        "mean_ocr_confidence": _mean([row["mean_ocr_confidence"] for row in rows]),
        "mean_duration_seconds": _mean([row["duration_seconds"] for row in rows]),
        "entity_precision": entity_tp / entity_predicted if entity_predicted else 0.0,
        "entity_recall": entity_tp / entity_expected if entity_expected else 0.0,
        "entity_f1": _f1(entity_tp, entity_predicted, entity_expected),
        "relation_precision": relation_tp / relation_predicted if relation_predicted else 0.0,
        "relation_recall": relation_tp / relation_expected if relation_expected else 0.0,
        "relation_f1": _f1(relation_tp, relation_predicted, relation_expected),
        "field_accuracy": sum(row["field_correct"] for row in rows) / max(1, sum(row["field_applicable"] for row in rows)),
        "document_type_accuracy": sum(row["document_type_correct"] for row in rows) / len(rows),
        "orientation_exact_accuracy": sum(bool(row["orientation_exact"]) for row in rows) / len(rows),
        "mean_orientation_error": _mean([row["orientation_error"] for row in rows]),
    }


def _f1(tp: int, predicted: int, expected: int) -> float:
    precision = tp / predicted if predicted else 0.0
    recall = tp / expected if expected else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _mean(values: list[Any]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else None


def _circular_error(left: float, right: float) -> float:
    delta = abs((left - right) % 360.0)
    return min(delta, 360.0 - delta)


def _angles(value: str) -> list[float]:
    result = []
    for item in value.split(","):
        angle = float(item) % 360.0
        if angle not in result:
            result.append(angle)
    if not result:
        raise ValueError("at least one angle is required")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
