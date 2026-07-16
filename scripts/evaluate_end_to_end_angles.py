#!/usr/bin/env python3
"""Run bounded public and synthetic-Thai end-to-end extraction at required angles."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from src import config as cfgmod  # noqa: E402
from src.evaluation.metrics import extraction_metrics, ocr_text_metrics, text_detection_metrics  # noqa: E402
from src.information_extraction.geometry import rotate_image_and_annotation  # noqa: E402
from src.inference.document_io import DocumentPage  # noqa: E402
from src.inference.document_pipeline import DocumentPipeline  # noqa: E402
from src.ocr.environment import configure_external_environment, require_storage_gate  # noqa: E402
from src.rotation_common import atomic_write_json, deterministic_rank, read_csv_rows, sha256_file  # noqa: E402

ANGLES = (0, 1, 15, 30, 37, 45, 60, 89, 90, 91, 135, 179, 180, 225, 269, 270, 315, 359)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", choices=("cpu", "gpu:0"), default="gpu:0")
    parser.add_argument("--pages-per-dataset", type=int, default=1)
    parser.add_argument("--model-setup", default=str(PROJECT_ROOT / "reports" / "ocr" / "model_setup.json"))
    args = parser.parse_args()
    if args.pages_per_dataset < 1:
        parser.error("--pages-per-dataset must be positive")
    cfg = cfgmod.load_config(args.config)
    asset_root = cfgmod.resolve_path(cfg, "external_assets")
    configure_external_environment(asset_root)
    require_storage_gate(
        asset_root, operation="end-to-end fixed-angle evaluation",
        anticipated_c_gib=0.25, anticipated_asset_gib=8.0,
    )
    checkpoint = Path(args.checkpoint).resolve()
    calibration = cfgmod.project_root(cfg) / "models" / "multitask_calibration.json"
    samples = _public_samples(cfg, args.pages_per_dataset)
    pipeline = DocumentPipeline.from_config(
        cfg, device=args.device, model_setup=args.model_setup,
        layout_checkpoint=checkpoint, calibration_path=calibration,
        enable_kmeans_display=False,
        require_layout_model=True,
    )
    started = time.perf_counter()
    public_observations = []
    thai_observations = []
    try:
        for angle in ANGLES:
            angle_rows = []
            for sample in samples:
                with Image.open(PROJECT_ROOT / sample["image_path"]) as source:
                    image = source.convert("RGB")
                annotation = json.loads(
                    (PROJECT_ROOT / sample["normalized_annotation_path"]).read_text(encoding="utf-8")
                )
                rotated_image, rotated_annotation, _ = rotate_image_and_annotation(
                    image, annotation, float(angle)
                )
                result = pipeline.extract_pages(
                    document_id=f"public_angle_{angle}", source_type="image",
                    pages=[DocumentPage(1, rotated_image)], language="auto",
                )
                page = result["pages"][0]
                reference_text = " ".join(str(token.get("text", "")) for token in rotated_annotation.get("tokens", []))
                text = ocr_text_metrics(reference_text, page["full_text"])
                detection = text_detection_metrics(
                    rotated_annotation.get("tokens", []), page["ocr"].get("words", [])
                )
                extraction = extraction_metrics(rotated_annotation, page, result["fields"])
                angle_rows.append({
                    "dataset": sample["dataset"],
                    "recognized_text_coverage": text["recognized_text_coverage"],
                    "wer": text["wer"],
                    "detection_f1": detection["f1"],
                    "entity_f1": extraction["entity"]["f1"],
                    "entity_expected": extraction["entity"]["expected"],
                    "relation_f1": extraction["relation"]["f1"],
                    "relation_expected": extraction["relation"]["expected"],
                    "field_correct": extraction["canonical_fields"]["correct"],
                    "field_applicable": extraction["canonical_fields"]["applicable"],
                    "entity_count": len(page["entities"]),
                    "table_count": len(page.get("tables") or []),
                    "nonempty": bool(page["full_text"].strip()),
                    "route": page["ocr"]["language_route"],
                    "selected_orientation": page["selected_ocr_orientation"],
                })
            public_observations.append(_aggregate_angle(angle, angle_rows))

            thai_image, thai_reference = _synthetic_thai_page()
            thai_annotation = {"page": {"width": thai_image.width, "height": thai_image.height}, "tokens": [], "entities": []}
            rotated_thai, _, _ = rotate_image_and_annotation(thai_image, thai_annotation, float(angle))
            thai_result = pipeline.extract_pages(
                document_id=f"synthetic_thai_angle_{angle}", source_type="image",
                pages=[DocumentPage(1, rotated_thai)], language="auto", language_hint="th",
            )
            thai_page = thai_result["pages"][0]
            thai_text = ocr_text_metrics(thai_reference, thai_page["full_text"])
            thai_observations.append({
                "angle": angle,
                "recognized_text_coverage": thai_text["recognized_text_coverage"],
                "wer": thai_text["wer"],
                "nonempty": bool(thai_page["full_text"].strip()),
                "route": thai_page["ocr"]["language_route"],
                "entity_count": len(thai_page["entities"]),
            })
    finally:
        pipeline.close()

    baseline = public_observations[0]
    for item in public_observations:
        item["extraction_retention_vs_upright"] = (
            item["entity_f1"] / baseline["entity_f1"] if baseline["entity_f1"] > 0 else None
        )
    report = {
        "schema_version": "1.0",
        "profile": "final",
        "split": "test_in_domain",
        "public_only_selection": True,
        "private_document_count": 0,
        "checkpoint": str(checkpoint),
        "checkpoint_model_sha256": sha256_file(checkpoint / "model.safetensors"),
        "calibration_sha256": sha256_file(calibration),
        "angles": list(ANGLES),
        "public_sample_pages": len(samples),
        "public_datasets": sorted({sample["dataset"] for sample in samples}),
        "public_metrics": public_observations,
        "synthetic_thai_metrics": thai_observations,
        "kmeans_controls_ocr": False,
        "duration_seconds": time.perf_counter() - started,
        "limitations": [
            "The expensive end-to-end angle grid is a deterministic bounded public sample; full test-head metrics are reported separately.",
            "Thai angle coverage uses a local synthetic page because the public held-out corpus has no compatible Thai polygon annotations.",
        ],
    }
    output = cfgmod.resolve_path(cfg, "reports") / "final_model" / "end_to_end_angle_metrics.json"
    atomic_write_json(output, report)
    print(json.dumps(report, indent=2))
    return 0


def _public_samples(cfg: dict[str, Any], limit: int) -> list[dict[str, str]]:
    rows = [
        row for row in read_csv_rows(cfgmod.resolve_path(cfg, "metadata") / "information_extraction_split_manifest.csv")
        if row.get("project_split") == "test_in_domain"
        and row.get("dataset") in {"fatura", "funsd", "sroie"}
        and row.get("is_private") == "false"
    ]
    buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        buckets[row["dataset"]].append(row)
    selected = []
    for dataset in sorted(buckets):
        selected.extend(sorted(
            buckets[dataset], key=lambda row: deterministic_rank(row["page_id"], 9090)
        )[:limit])
    return selected


def _aggregate_angle(angle: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "angle": angle,
        **_aggregate_rows(rows),
        "by_dataset": {
            dataset: _aggregate_rows([
                row for row in rows if row["dataset"] == dataset
            ])
            for dataset in sorted({row["dataset"] for row in rows})
        },
    }


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    field_applicable = sum(row["field_applicable"] for row in rows)
    detections = [row["detection_f1"] for row in rows if row["detection_f1"] is not None]
    return {
        "page_count": len(rows),
        "recognized_text_coverage": statistics.fmean(row["recognized_text_coverage"] for row in rows),
        "wer": statistics.fmean(row["wer"] for row in rows),
        "detection_f1": statistics.fmean(detections) if detections else None,
        "entity_f1": statistics.fmean(row["entity_f1"] for row in rows),
        "relation_f1": statistics.fmean(row["relation_f1"] for row in rows),
        "field_accuracy": sum(row["field_correct"] for row in rows) / field_applicable if field_applicable else None,
        "nonempty_rate": sum(row["nonempty"] for row in rows) / len(rows),
        "route_counts": {
            route: sum(row["route"] == route for row in rows)
            for route in sorted({row["route"] for row in rows})
        },
        "error_counts": {
            "ocr_coverage_below_0_70": sum(row["recognized_text_coverage"] < 0.70 for row in rows),
            "detection_f1_below_0_70": sum(
                row["detection_f1"] is not None and row["detection_f1"] < 0.70 for row in rows
            ),
            "entity_f1_below_0_70_with_reference": sum(
                row["entity_expected"] > 0 and row["entity_f1"] < 0.70 for row in rows
            ),
            "relation_f1_below_0_60_with_reference": sum(
                row["relation_expected"] > 0 and row["relation_f1"] < 0.60 for row in rows
            ),
            "canonical_field_miss": sum(
                row["field_applicable"] > row["field_correct"] for row in rows
            ),
            "no_table_detected": sum(row["table_count"] == 0 for row in rows),
            "empty_ocr": sum(not row["nonempty"] for row in rows),
        },
    }


def _synthetic_thai_page() -> tuple[Image.Image, str]:
    text = "ใบเสร็จ ยอดรวม 123.45 บาท"
    image = Image.new("RGB", (900, 220), "white")
    font_path = Path("C:/Windows/Fonts/tahoma.ttf")
    font = ImageFont.truetype(str(font_path), 52) if font_path.is_file() else ImageFont.load_default()
    ImageDraw.Draw(image).text((35, 70), text, fill="black", font=font)
    return image, text


if __name__ == "__main__":
    raise SystemExit(main())
