#!/usr/bin/env python3
"""Extract structured information from local images or PDFs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config as cfgmod  # noqa: E402
from src.inference.document_io import load_document_pages  # noqa: E402
from src.inference.document_pipeline import DocumentPipeline  # noqa: E402
from src.inference.output_writer import (  # noqa: E402
    require_private_output_root,
    write_document_outputs,
)
from src.ocr.environment import configure_external_environment  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, nargs="+", help="one or more image/PDF files")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--language", choices=("auto", "general", "thai", "en", "tr", "th"), default="auto")
    parser.add_argument("--language-hint")
    parser.add_argument("--device", choices=("cpu", "gpu:0"), default="cpu")
    parser.add_argument("--disable-kmeans-display", action="store_true")
    parser.add_argument("--save-ocr-visualization", action="store_true")
    parser.add_argument("--private-output", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-pages", type=int)
    parser.add_argument("--continue-on-page-error", action="store_true")
    parser.add_argument("--continue-on-document-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pdf-dpi", type=int, default=200)
    parser.add_argument("--deskew-angle", type=float, help="optional evidence-backed correction candidate")
    parser.add_argument("--layout-checkpoint")
    parser.add_argument("--model-setup", default=str(PROJECT_ROOT / "reports" / "ocr" / "model_setup.json"))
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    args = parser.parse_args()
    if args.max_pages is not None and args.max_pages < 1:
        parser.error("--max-pages must be positive")
    cfg = cfgmod.load_config(args.config)
    configure_external_environment(cfgmod.resolve_path(cfg, "external_assets"))
    inputs = [Path(value) for value in args.input]
    output = Path(args.output)
    if args.private_output:
        require_private_output_root(output, cfgmod.resolve_path(cfg, "private_outputs"))
    if args.dry_run:
        checks = []
        for source in inputs:
            document_id, source_type, pages = load_document_pages(
                source, max_pages=args.max_pages, pdf_dpi=args.pdf_dpi
            )
            checks.append({"document_id": document_id, "source_type": source_type, "page_count": len(pages)})
        print(json.dumps({"status": "dry_run", "inputs": checks}, indent=2))
        return 0
    pipeline = DocumentPipeline.from_config(
        cfg,
        device=args.device,
        model_setup=args.model_setup,
        layout_checkpoint=args.layout_checkpoint,
        enable_kmeans_display=not args.disable_kmeans_display,
    )
    failures = []
    outputs = []
    try:
        for source in inputs:
            try:
                document_id, source_type, pages = load_document_pages(
                    source, max_pages=args.max_pages, pdf_dpi=args.pdf_dpi
                )
                result = pipeline.extract_pages(
                    document_id=document_id,
                    source_type=source_type,
                    pages=pages,
                    language=args.language,
                    language_hint=args.language_hint,
                    private_output=args.private_output,
                    continue_on_page_error=args.continue_on_page_error,
                    deskew_angle=args.deskew_angle,
                )
                destination = output if len(inputs) == 1 else output / document_id
                result_path = write_document_outputs(
                    result,
                    pages,
                    destination,
                    force=args.force,
                    save_visualization=args.save_ocr_visualization,
                )
                outputs.append(str(result_path))
            except Exception as exc:
                failures.append({"input_index": len(outputs) + len(failures), "error": f"{type(exc).__name__}: {exc}"})
                if not args.continue_on_document_error:
                    print(json.dumps({"status": "failed", "failures": failures}, indent=2), file=sys.stderr)
                    return 1
    finally:
        pipeline.close()
    print(json.dumps({"status": "complete" if not failures else "partial", "outputs": outputs, "failures": failures}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
