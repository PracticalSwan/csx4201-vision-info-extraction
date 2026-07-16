#!/usr/bin/env python3
"""Internal JSON-lines worker for isolated LayoutXLM inference."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ocr.environment import configure_external_environment  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cache-dir")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--calibration")
    parser.add_argument("--confidence-threshold", type=float)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    configure_external_environment()
    from src.information_extraction.multitask_inference import MultiTaskLayoutExtractor

    extractor = MultiTaskLayoutExtractor(
        args.checkpoint,
        device=args.device,
        cache_dir=args.cache_dir,
        max_length=args.max_length,
        calibration_path=args.calibration,
        confidence_threshold=args.confidence_threshold,
    )
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("action") == "shutdown":
                return 0
            if request.get("action") != "extract":
                raise ValueError("unsupported worker action")
            result = extractor.extract(
                request["ocr_result"],
                page_number=int(request["page_number"]),
                width=int(request["width"]),
                height=int(request["height"]),
            )
            response = {"status": "ok", **result}
        except Exception as exc:
            response = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(response, ensure_ascii=False, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
