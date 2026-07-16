#!/usr/bin/env python3
"""Verify exact OCR artifacts plus Thai, cardinal, and arbitrary-angle OCR."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

from src.ocr.environment import DEFAULT_ASSET_ROOT, configure_external_environment  # noqa: E402
from src.ocr.model_registry import ModelRegistry  # noqa: E402
from src.ocr.paddleocr_adapter import PaddleOCRAdapter  # noqa: E402
from src.ocr.pipeline import MultilingualOCR  # noqa: E402
from src.rotation_common import atomic_write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=("cpu", "gpu:0"))
    parser.add_argument("--asset-root", default=str(DEFAULT_ASSET_ROOT))
    parser.add_argument(
        "--setup", default=str(PROJECT_ROOT / "reports" / "ocr" / "model_setup.json")
    )
    parser.add_argument(
        "--output", default=str(PROJECT_ROOT / "reports" / "ocr" / "model_verification.json")
    )
    args = parser.parse_args()
    configure_external_environment(args.asset_root)
    started = time.perf_counter()
    registry = ModelRegistry.from_setup(args.setup, verify_hashes=True)
    general = PaddleOCRAdapter(registry, "general", device=args.device)
    thai = PaddleOCRAdapter(registry, "thai", device=args.device)
    rotation_pipeline = MultilingualOCR(
        general_backend=general,
        thai_backend=thai,
    )
    tests = {
        "general": _smoke(general, _synthetic("INVOICE TOTAL 123.45")),
        "thai": _smoke(thai, _synthetic("ใบเสร็จรับเงิน ยอดรวม 123.45")),
        "rotated_general": _rotation_smoke(
            rotation_pipeline,
            _synthetic("INVOICE TOTAL 123.45").rotate(
                90, expand=True, fillcolor="white"
            ),
        ),
        "arbitrary_angle_general": _arbitrary_rotation_smoke(
            rotation_pipeline,
            _synthetic("INVOICE TOTAL 123.45").rotate(
                17, expand=True, fillcolor="white"
            ),
        ),
    }
    payload = {
        "schema_version": "1.0", "device": args.device,
        "artifact_hash_verification": "passed",
        "tests": tests, "passed": all(item["passed"] for item in tests.values()),
        "duration_seconds": time.perf_counter() - started,
    }
    atomic_write_json(Path(args.output), payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["passed"] else 1


def _smoke(adapter: PaddleOCRAdapter, image: Image.Image) -> dict:
    try:
        result = adapter.predict(image)
        valid = (
            result["detector_model"] == adapter.detector.name
            and result["recognizer_model"] == adapter.recognizer.name
            and isinstance(result["words"], list)
            and bool(result["words"])
        )
        return {
            "passed": valid, "word_count": len(result["words"]),
            "mean_confidence": result["mean_confidence"],
            "detector_model": result["detector_model"], "recognizer_model": result["recognizer_model"],
            "recognized_text_preview": result["full_text"][:120],
        }
    except Exception as exc:
        return {"passed": False, "error": f"{type(exc).__name__}: {exc}"}


def _rotation_smoke(pipeline: MultilingualOCR, image: Image.Image) -> dict:
    """Require the cardinal selector to recover the known rotated phrase."""
    try:
        result = pipeline.extract_page(image, language_mode="general")
        normalized = " ".join(str(result.get("full_text", "")).upper().split())
        required_fragments = ("INVOICE", "TOTAL", "123.45")
        valid = (
            result.get("detector_model") == "PP-OCRv6_medium_det"
            and result.get("recognizer_model") == "PP-OCRv6_medium_rec"
            and bool(result.get("words"))
            and all(fragment in normalized for fragment in required_fragments)
            and float(result.get("orientation", 0.0)) % 360.0 != 0.0
        )
        return {
            "passed": valid,
            "word_count": len(result.get("words") or []),
            "mean_confidence": result.get("mean_confidence"),
            "detector_model": result.get("detector_model"),
            "recognizer_model": result.get("recognizer_model"),
            "selected_orientation": result.get("orientation"),
            "required_fragments": list(required_fragments),
            "recognized_text_preview": str(result.get("full_text", ""))[:120],
        }
    except Exception as exc:
        return {"passed": False, "error": f"{type(exc).__name__}: {exc}"}


def _arbitrary_rotation_smoke(pipeline: MultilingualOCR, image: Image.Image) -> dict:
    """Require a polygon-derived fine candidate and real phrase recovery."""
    try:
        result = pipeline.extract_page(image, language_mode="general")
        normalized = " ".join(str(result.get("full_text", "")).upper().split())
        required_fragments = ("INVOICE", "TOTAL", "123.45")
        fine_scores = [
            value
            for value in result.get("candidate_scores", [])
            if value.get("candidate_kind") == "fine_deskew"
        ]
        valid = (
            result.get("detector_model") == "PP-OCRv6_medium_det"
            and result.get("recognizer_model") == "PP-OCRv6_medium_rec"
            and bool(result.get("words"))
            and all(fragment in normalized for fragment in required_fragments)
            and bool(fine_scores)
            and result.get("fine_deskew") is not None
        )
        return {
            "passed": valid,
            "word_count": len(result.get("words") or []),
            "mean_confidence": result.get("mean_confidence"),
            "detector_model": result.get("detector_model"),
            "recognizer_model": result.get("recognizer_model"),
            "selected_orientation": result.get("orientation"),
            "fine_candidate_count": len(fine_scores),
            "selected_fine_deskew": result.get("fine_deskew"),
            "required_fragments": list(required_fragments),
            "recognized_text_preview": str(result.get("full_text", ""))[:120],
        }
    except Exception as exc:
        return {"passed": False, "error": f"{type(exc).__name__}: {exc}"}


def _synthetic(text: str) -> Image.Image:
    image = Image.new("RGB", (1200, 260), "white")
    draw = ImageDraw.Draw(image)
    font = _font(72)
    draw.text((30, 70), text, fill="black", font=font)
    return image


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        Path("C:/Windows/Fonts/tahoma.ttf"), Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
    ):
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


if __name__ == "__main__":
    raise SystemExit(main())
