#!/usr/bin/env python3
"""Run read-only local inference on private documents with anonymous outputs."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config as cfgmod  # noqa: E402
from src.inference.document_io import load_document_pages  # noqa: E402
from src.inference.document_pipeline import DocumentPipeline  # noqa: E402
from src.inference.output_writer import require_private_output_root, write_document_outputs  # noqa: E402
from src.inference.private_evaluation import (  # noqa: E402
    aggregate_markdown,
    aggregate_private_results,
    anonymous_document_id,
    discover_private_documents,
    manual_review_rows,
)
from src.ocr.environment import configure_external_environment, require_storage_gate  # noqa: E402
from src.rotation_common import atomic_write_csv, atomic_write_json, atomic_write_text, sha256_file  # noqa: E402

MANUAL_REVIEW_COLUMNS = (
    "anonymous_document_id",
    "predicted_document_type",
    "extracted_field",
    "predicted_value",
    "confidence",
    "evidence_page",
    "user_corrected_value",
    "correct_yes_no",
    "notes",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--language", choices=("auto", "general", "en", "tr", "thai", "th"), default="auto")
    parser.add_argument("--device", choices=("cpu", "gpu:0"), default="gpu:0")
    parser.add_argument("--private-output", action="store_true")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--file", action="append", default=[])
    parser.add_argument("--recursive", action="store_true")
    visualizations = parser.add_mutually_exclusive_group()
    visualizations.add_argument("--save-private-visualizations", action="store_true")
    visualizations.add_argument("--no-private-visualizations", action="store_true")
    parser.add_argument(
        "--aggregate-report", nargs="?", const="reports/final_model/private_test_aggregate.json"
    )
    parser.add_argument("--manual-review-csv", nargs="?", const="manual_review.csv")
    parser.add_argument("--max-pages", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--model-setup", default=str(PROJECT_ROOT / "reports" / "ocr" / "model_setup.json"))
    args = parser.parse_args()
    if not args.private_output:
        parser.error("--private-output is mandatory for private testing")
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.max_pages is not None and args.max_pages < 1:
        parser.error("--max-pages must be positive")

    cfg = cfgmod.load_config(args.config)
    asset_root = cfgmod.resolve_path(cfg, "external_assets")
    configure_external_environment(asset_root)
    require_storage_gate(
        asset_root,
        operation="private operational inference",
        anticipated_c_gib=0.25,
        anticipated_asset_gib=5.0,
    )
    output_root = Path(args.output_root)
    private_root = cfgmod.resolve_path(cfg, "private_outputs")
    require_private_output_root(output_root, private_root)
    sources = discover_private_documents(
        args.input_root,
        explicit_files=args.file,
        recursive=args.recursive,
        limit=args.limit,
    )
    if not sources:
        raise SystemExit("no supported private images or PDFs were selected")

    checkpoint = Path(args.checkpoint)
    model_file = checkpoint / "model.safetensors"
    model_hash = sha256_file(model_file) if model_file.is_file() else None
    pipeline = DocumentPipeline.from_config(
        cfg,
        device=args.device,
        model_setup=args.model_setup,
        layout_checkpoint=checkpoint,
        enable_kmeans_display=False,
        require_layout_model=True,
    )
    results = []
    review_rows = []
    errors: Counter[str] = Counter()
    started = time.perf_counter()
    try:
        for index, source in enumerate(sources, start=1):
            anonymous_id = anonymous_document_id(index)
            destination = output_root / "documents" / anonymous_id
            try:
                _, source_type, pages = load_document_pages(source, max_pages=args.max_pages)
                result = pipeline.extract_pages(
                    document_id=anonymous_id,
                    source_type=source_type,
                    pages=pages,
                    language=args.language,
                    private_output=True,
                    continue_on_page_error=args.continue_on_error,
                )
                write_document_outputs(
                    result,
                    pages,
                    destination,
                    force=args.force,
                    save_visualization=args.save_private_visualizations,
                )
                results.append(result)
                review_rows.extend(manual_review_rows(anonymous_id, result))
            except Exception as exc:
                errors[type(exc).__name__] += 1
                atomic_write_json(destination / "error.json", {
                    "anonymous_document_id": anonymous_id,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                })
                if not args.continue_on_error:
                    break
    finally:
        pipeline.close()

    report = aggregate_private_results(
        results,
        attempted_documents=len(results) + sum(errors.values()),
        error_type_counts=errors,
        elapsed_seconds=time.perf_counter() - started,
        checkpoint_model_sha256=model_hash,
    )
    atomic_write_json(output_root / "aggregate_report.json", report)
    atomic_write_text(output_root / "aggregate_report.md", aggregate_markdown(report))
    review_path = output_root / (args.manual_review_csv or "manual_review.csv")
    require_private_output_root(review_path, private_root)
    atomic_write_csv(review_path, review_rows, MANUAL_REVIEW_COLUMNS)
    if args.aggregate_report:
        public_report = Path(args.aggregate_report)
        if not public_report.is_absolute():
            public_report = PROJECT_ROOT / public_report
        atomic_write_json(public_report, report)
        atomic_write_text(public_report.with_suffix(".md"), aggregate_markdown(report))
    print(json.dumps({
        "status": "complete" if not errors else "partial",
        "attempted_documents": report["attempted_documents"],
        "successful_documents": report["successful_documents"],
        "failed_documents": report["failed_documents"],
        "output_root": str(output_root.resolve()),
    }, indent=2))
    return 0 if report["successful_documents"] and not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
