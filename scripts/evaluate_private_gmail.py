#!/usr/bin/env python3
"""Run local Gmail private-test inference and write aggregate-only metrics."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image  # noqa: E402

from src import config as cfgmod  # noqa: E402
from src.inference.document_io import DocumentPage  # noqa: E402
from src.inference.document_pipeline import DocumentPipeline  # noqa: E402
from src.ocr.environment import configure_external_environment  # noqa: E402
from src.rotation_common import atomic_write_json, deterministic_rank, read_csv_rows  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "gpu:0"), default="gpu:0")
    parser.add_argument("--layout-checkpoint")
    parser.add_argument("--model-setup", default=str(PROJECT_ROOT / "reports" / "ocr" / "model_setup.json"))
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    cfg = cfgmod.load_config(args.config)
    configure_external_environment(cfgmod.resolve_path(cfg, "external_assets"))
    private_manifest = cfgmod.resolve_path(cfg, "metadata") / "private_information_extraction_manifest.csv"
    rows = [
        row for row in read_csv_rows(private_manifest)
        if row.get("is_private") == "true" and row.get("project_split") == "private_test"
        and row.get("is_usable") == "true" and row.get("image_path")
    ]
    rows.sort(key=lambda row: deterministic_rank(row["page_id"], 42))
    rows = rows[: args.limit]
    pipeline = DocumentPipeline.from_config(
        cfg,
        device=args.device,
        model_setup=args.model_setup,
        layout_checkpoint=args.layout_checkpoint,
        enable_kmeans_display=False,
    )
    counts: Counter[str] = Counter()
    confidences: list[float] = []
    durations: list[float] = []
    error_types: Counter[str] = Counter()
    started = time.perf_counter()
    try:
        for row in rows:
            try:
                with Image.open(PROJECT_ROOT / row["image_path"]) as source:
                    image = source.convert("RGB")
                result = pipeline.extract_pages(
                    document_id="private_document",
                    source_type="image",
                    pages=[DocumentPage(1, image)],
                    language="auto",
                    private_output=True,
                )
                page = result["pages"][0]
                counts["successful_pages"] += 1
                counts["ocr_words"] += len(page["ocr"]["words"])
                counts["entities"] += len(page["entities"])
                counts["relations"] += len(page["key_value_pairs"])
                counts["non_null_fields"] += sum(value is not None for value in result["fields"].values())
                counts[f"route:{page['ocr']['language_route']}"] += 1
                if page["ocr"]["mean_confidence"] is not None:
                    confidences.append(float(page["ocr"]["mean_confidence"]))
                durations.append(float(result["processing"]["duration_seconds"]))
            except Exception as exc:
                counts["failed_pages"] += 1
                error_types[type(exc).__name__] += 1
    finally:
        pipeline.close()
    attempted = len(rows)
    report = {
        "schema_version": "1.0",
        "status": "private_test_aggregate",
        "attempted_pages": attempted,
        "successful_pages": counts["successful_pages"],
        "failed_pages": counts["failed_pages"],
        "mean_ocr_words": counts["ocr_words"] / max(1, counts["successful_pages"]),
        "mean_entities": counts["entities"] / max(1, counts["successful_pages"]),
        "mean_relations": counts["relations"] / max(1, counts["successful_pages"]),
        "mean_non_null_fields": counts["non_null_fields"] / max(1, counts["successful_pages"]),
        "mean_ocr_confidence": sum(confidences) / len(confidences) if confidences else None,
        "mean_duration_seconds": sum(durations) / len(durations) if durations else None,
        "route_counts": {key.split(":", 1)[1]: value for key, value in counts.items() if key.startswith("route:")},
        "error_type_counts": dict(error_types),
        "gmail_fit_rows": 0,
        "local_processing_only": True,
        "contains_filenames": False,
        "contains_ocr_text": False,
        "contains_images": False,
        "contains_per_document_predictions": False,
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": ["No private ground truth is used; this is operational aggregate testing, not accuracy evaluation."],
    }
    atomic_write_json(
        cfgmod.resolve_path(cfg, "reports") / "model_evaluation" / "private_gmail_aggregate.json",
        report,
    )
    print(json.dumps(report, indent=2))
    return 0 if attempted and counts["successful_pages"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
