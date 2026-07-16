#!/usr/bin/env python3
"""Run a bounded public dev-select OCR preprocessing and module ablation."""
from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image  # noqa: E402

from src import config as cfgmod  # noqa: E402
from src.information_extraction.alignment import align_ocr_to_annotations, normalize_alignment_text  # noqa: E402
from src.ocr.cache import OCRCache  # noqa: E402
from src.ocr.environment import configure_external_environment, require_storage_gate  # noqa: E402
from src.ocr.model_registry import ModelRegistry  # noqa: E402
from src.ocr.pipeline import MultilingualOCR  # noqa: E402
from src.rotation_common import atomic_write_json, deterministic_rank, read_csv_rows  # noqa: E402


SETTINGS = (
    ("original", "original", {}),
    ("grayscale_normalized", "grayscale_normalized", {}),
    ("adaptive_contrast", "adaptive_contrast", {}),
    ("denoise", "denoise", {}),
    ("sharpen", "sharpen", {}),
    ("background_normalized", "background_normalized", {}),
    ("quality_auto", "quality_auto", {}),
    ("doc_orientation_on", "original", {"use_doc_orientation_classify": True}),
    ("textline_orientation_on", "original", {"use_textline_orientation": True}),
    ("unwarping_on", "original", {"use_doc_unwarping": True}),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--device", choices=("cpu", "gpu:0"), default="gpu:0")
    parser.add_argument("--limit-per-dataset", type=int, default=1)
    parser.add_argument("--model-setup", default=str(PROJECT_ROOT / "reports" / "ocr" / "model_setup.json"))
    args = parser.parse_args()
    if args.limit_per_dataset < 1:
        parser.error("--limit-per-dataset must be positive")
    cfg = cfgmod.load_config(args.config)
    asset_root = cfgmod.resolve_path(cfg, "external_assets")
    configure_external_environment(asset_root)
    require_storage_gate(
        asset_root,
        operation="public OCR preprocessing ablation",
        anticipated_c_gib=0.25,
        anticipated_asset_gib=5.0,
    )
    registry = ModelRegistry.from_setup(args.model_setup)
    cache = OCRCache(cfgmod.resolve_path(cfg, "ocr_cache"))
    rows = _selected_rows(cfg, args.limit_per_dataset)
    started = time.perf_counter()
    setting_reports = []
    for name, profile, options in SETTINGS:
        try:
            pipeline = MultilingualOCR(
                registry,
                device=args.device,
                adapter_options=options,
                cache=cache,
                preprocessing_version="2.1-quality-profile-ablation",
                preprocessing_profile=profile,
            )
            observations = [_evaluate_row(cfg, pipeline, row) for row in rows]
            setting_reports.append(_aggregate_setting(name, profile, options, observations))
            del pipeline
        except Exception as exc:
            setting_reports.append({
                "name": name,
                "preprocessing_profile": profile,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
            })

    dpi_report = _dpi_ablation(cfg, registry, cache, rows[0], args.device)
    successful_settings = [
        item for item in setting_reports if item.get("status") == "passed"
    ]
    if not successful_settings:
        raise RuntimeError("all OCR preprocessing ablation settings failed")
    best = max(
        successful_settings,
        key=lambda item: (
            item["mean_alignment_coverage"],
            -item["mean_word_error_rate"],
            item["name"] == "original",
        ),
    )
    report = {
        "schema_version": "1.0",
        "selection_split": "dev_select",
        "public_only": True,
        "private_page_count": 0,
        "sample_pages": len(rows),
        "limit_per_dataset": args.limit_per_dataset,
        "datasets": sorted({row["dataset"] for row in rows}),
        "settings": setting_reports,
        "dpi_ablation": dpi_report,
        "chosen_setting": best["name"],
        "chosen_preprocessing_profile": best["preprocessing_profile"],
        "selection_metric": "maximize mean token-alignment coverage, then minimize word error rate",
        "duration_seconds": time.perf_counter() - started,
        "limitations": [
            "This is a bounded dev-select ablation; the locked public test and private data were not used for selection.",
            "The public corpus is raster-only, so DPI was tested by wrapping one selected public raster page in a derived PDF before rendering at 200, 250, and 300 DPI.",
        ],
    }
    output = cfgmod.resolve_path(cfg, "reports") / "final_model" / "ocr_preprocessing_ablation.json"
    atomic_write_json(output, report)
    print(json.dumps(report, indent=2))
    return 0


def _selected_rows(cfg: dict[str, Any], limit: int) -> list[dict[str, str]]:
    metadata = cfgmod.resolve_path(cfg, "metadata")
    rows = [
        row for row in read_csv_rows(metadata / "information_extraction_split_manifest.csv")
        if row.get("project_split") == "dev_select"
        and row.get("dataset") in {"fatura", "funsd", "sroie"}
        and row.get("is_private") == "false"
    ]
    buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        buckets[row["dataset"]].append(row)
    selected = []
    for dataset in sorted(buckets):
        ordered = sorted(buckets[dataset], key=lambda row: deterministic_rank(row["page_id"], 2026))
        selected.extend(ordered[:limit])
    return selected


def _evaluate_row(cfg: dict[str, Any], pipeline: MultilingualOCR, row: dict[str, str]) -> dict[str, Any]:
    annotation = json.loads((PROJECT_ROOT / row["normalized_annotation_path"]).read_text(encoding="utf-8"))
    route = "thai" if row.get("language") == "th" else "general"
    result = pipeline.extract_path(PROJECT_ROOT / row["image_path"], language_mode=route)
    return _score_result(annotation, result, dataset=row["dataset"])


def _score_result(annotation: dict[str, Any], result: dict[str, Any], *, dataset: str) -> dict[str, Any]:
    reference = [normalize_alignment_text(str(token.get("text", ""))) for token in annotation.get("tokens", [])]
    hypothesis = [normalize_alignment_text(str(word.get("text", ""))) for word in result.get("words", [])]
    reference = [value for value in reference if value]
    hypothesis = [value for value in hypothesis if value]
    alignment = align_ocr_to_annotations(result.get("words", []), annotation.get("tokens", []))
    return {
        "dataset": dataset,
        "alignment_coverage": float(alignment["alignment_coverage"]),
        "word_error_rate": word_error_rate(reference, hypothesis),
        "mean_confidence": float(result.get("mean_confidence", 0.0) or 0.0),
        "word_count": len(hypothesis),
        "selected_orientation": float(result.get("orientation", 0.0)),
    }


def _aggregate_setting(name: str, profile: str, options: dict[str, Any], observations: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in observations:
        grouped[item["dataset"]].append(item)
    return {
        "name": name,
        "status": "passed",
        "preprocessing_profile": profile,
        "paddle_modules": {
            "use_doc_orientation_classify": bool(options.get("use_doc_orientation_classify", False)),
            "use_doc_unwarping": bool(options.get("use_doc_unwarping", False)),
            "use_textline_orientation": bool(options.get("use_textline_orientation", False)),
        },
        "page_count": len(observations),
        "mean_alignment_coverage": statistics.fmean(item["alignment_coverage"] for item in observations),
        "mean_word_error_rate": statistics.fmean(item["word_error_rate"] for item in observations),
        "mean_confidence": statistics.fmean(item["mean_confidence"] for item in observations),
        "mean_word_count": statistics.fmean(item["word_count"] for item in observations),
        "by_dataset": {
            dataset: {
                "page_count": len(items),
                "mean_alignment_coverage": statistics.fmean(item["alignment_coverage"] for item in items),
                "mean_word_error_rate": statistics.fmean(item["word_error_rate"] for item in items),
                "mean_confidence": statistics.fmean(item["mean_confidence"] for item in items),
            }
            for dataset, items in sorted(grouped.items())
        },
    }


def _dpi_ablation(
    cfg: dict[str, Any], registry: ModelRegistry, cache: OCRCache,
    row: dict[str, str], device: str,
) -> dict[str, Any]:
    import fitz

    annotation = json.loads((PROJECT_ROOT / row["normalized_annotation_path"]).read_text(encoding="utf-8"))
    with Image.open(PROJECT_ROOT / row["image_path"]) as source:
        buffer = io.BytesIO()
        source.convert("RGB").save(buffer, format="PDF", resolution=72.0)
    document = fitz.open(stream=buffer.getvalue(), filetype="pdf")
    try:
        page = document.load_page(0)
        observations = []
        pipeline = MultilingualOCR(
            registry, device=device, cache=cache,
            preprocessing_version="2.1-dpi-ablation", preprocessing_profile="original",
        )
        for dpi in (200, 250, 300):
            scale = dpi / 72.0
            pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            result = pipeline.extract_page(image, language_mode="general")
            score = _score_result(annotation, result, dataset=row["dataset"])
            observations.append({"dpi": dpi, "render_width": image.width, "render_height": image.height, **score})
    finally:
        document.close()
    best = max(observations, key=lambda item: (item["alignment_coverage"], -item["word_error_rate"], -item["dpi"]))
    return {
        "source": "derived public dev-select PDF from one raster page",
        "observations": observations,
        "chosen_dpi": best["dpi"],
    }


def word_error_rate(reference: list[str], hypothesis: list[str]) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    previous = list(range(len(hypothesis) + 1))
    for row, expected in enumerate(reference, start=1):
        current = [row]
        for column, actual in enumerate(hypothesis, start=1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (expected != actual),
            ))
        previous = current
    return previous[-1] / len(reference)


if __name__ == "__main__":
    raise SystemExit(main())
