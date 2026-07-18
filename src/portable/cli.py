"""One-command local CLI for the finished OCR information-extraction model."""
from __future__ import annotations

import argparse
import json
import sys

from .api import ExtractionError, run_extraction
from .results import field_rows
from .runtime import RuntimeSettings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run full local OCR + layout information extraction on one image or PDF."
    )
    parser.add_argument("input", help="image or PDF to extract")
    parser.add_argument("-o", "--output", help="output folder (default: timestamped outputs folder)")
    parser.add_argument(
        "--language",
        choices=("auto", "general", "thai", "en", "tr", "th"),
        default="auto",
    )
    parser.add_argument("--device", choices=("cpu", "gpu:0"))
    parser.add_argument("--max-pages", type=int)
    parser.add_argument("--no-visualization", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    try:
        runtime = RuntimeSettings.load()
        run = run_extraction(
            args.input,
            settings=runtime,
            output_dir=args.output,
            language=args.language,
            device=args.device,
            max_pages=args.max_pages,
            save_visualization=not args.no_visualization,
            on_log=None if args.quiet else lambda line: print(line, file=sys.stderr),
        )
    except (ExtractionError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "status": "complete",
                "result": str(run.result_path),
                "output_directory": str(run.output_dir),
                "fields": [
                    {
                        "name": row[0],
                        "value": row[1],
                        "confidence": row[2],
                    }
                    for row in field_rows(run.payload)
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0
