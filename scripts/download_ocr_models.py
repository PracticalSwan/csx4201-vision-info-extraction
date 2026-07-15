#!/usr/bin/env python3
"""Download and register the three exact required PaddleOCR models."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ocr.environment import DEFAULT_ASSET_ROOT, configure_external_environment  # noqa: E402
from src.ocr.model_registry import (  # noqa: E402
    DETECTOR_MODEL,
    GENERAL_RECOGNIZER_MODEL,
    REQUIRED_MODEL_NAMES,
    THAI_RECOGNIZER_MODEL,
)
from src.rotation_common import atomic_write_json, sha256_file  # noqa: E402

MODEL_METADATA = {
    DETECTOR_MODEL: {"role": "detector", "language": "multilingual"},
    GENERAL_RECOGNIZER_MODEL: {"role": "recognizer", "language": "general_50_languages"},
    THAI_RECOGNIZER_MODEL: {"role": "recognizer", "language": "thai_english_numbers"},
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="all", choices=("all", *REQUIRED_MODEL_NAMES))
    parser.add_argument("--device", default="cpu", choices=("cpu", "gpu:0"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--asset-root", default=str(DEFAULT_ASSET_ROOT))
    parser.add_argument(
        "--output", default=str(PROJECT_ROOT / "reports" / "ocr" / "model_setup.json")
    )
    args = parser.parse_args()
    asset_root = Path(args.asset_root)
    environment = configure_external_environment(asset_root)
    if args.offline:
        import os

        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    requested = list(REQUIRED_MODEL_NAMES) if args.model == "all" else [args.model]
    official_root = Path(environment["PADDLE_PDX_CACHE_HOME"]) / "official_models"
    official_root.mkdir(parents=True, exist_ok=True)
    existing_manifest = _load_existing(Path(args.output))
    records = dict(existing_manifest.get("models", {}))
    smoke_results: dict[str, Any] = dict(existing_manifest.get("component_smoke_tests", {}))
    for name in requested:
        path = official_root / name
        if args.force and path.exists():
            _safe_remove_model(path, official_root, name)
        if args.offline and not path.is_dir():
            raise SystemExit(f"offline model is unavailable: {name} at {path}")
        model, attempts = _initialize_with_retries(name, path if args.offline else None, args.device)
        resolved_name = str(getattr(model, "_model_name", ""))
        if resolved_name != name:
            raise SystemExit(f"model substitution detected: requested={name}, resolved={resolved_name}")
        if not path.is_dir():
            raise SystemExit(f"official API initialized {name} but expected cache path is absent: {path}")
        files = _file_manifest(path)
        if not files:
            raise SystemExit(f"model directory contains no files: {path}")
        smoke_results[name] = _component_smoke(model, name)
        records[name] = {
            "requested_name": name,
            "resolved_name": resolved_name,
            "resolved_path": str(path),
            "role": MODEL_METADATA[name]["role"],
            "language": MODEL_METADATA[name]["language"],
            "initialization_attempts": attempts,
            "device": args.device,
            "total_size_bytes": sum(item["size_bytes"] for item in files),
            "files": files,
        }
    missing = sorted(set(REQUIRED_MODEL_NAMES) - set(records))
    payload = {
        "schema_version": "1.0",
        "complete": not missing,
        "missing_models": missing,
        "python_version": sys.version.split()[0],
        "paddle_version": importlib.metadata.version("paddlepaddle-gpu")
        if _package_exists("paddlepaddle-gpu") else importlib.metadata.version("paddlepaddle"),
        "paddleocr_version": importlib.metadata.version("paddleocr"),
        "paddlex_version": importlib.metadata.version("paddlex"),
        "device": args.device,
        "asset_root": str(asset_root),
        "cache_root": str(official_root),
        "offline": bool(args.offline),
        "models": records,
        "component_smoke_tests": smoke_results,
    }
    atomic_write_json(Path(args.output), payload)
    print(json.dumps({
        "complete": payload["complete"], "missing_models": missing,
        "models": {name: {"path": item["resolved_path"], "bytes": item["total_size_bytes"]}
                   for name, item in records.items()},
        "output": args.output,
    }, indent=2))
    return 0 if not missing else 1


def _initialize_with_retries(name: str, model_dir: Path | None, device: str) -> tuple[Any, int]:
    from paddleocr import TextDetection, TextRecognition

    model_class = TextDetection if name == DETECTOR_MODEL else TextRecognition
    kwargs: dict[str, Any] = {"model_name": name, "device": device}
    if model_dir is not None:
        kwargs["model_dir"] = str(model_dir)
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            return model_class(**kwargs), attempt
        except Exception as exc:  # network/runtime errors vary by host
            last_error = exc
            if attempt < 3:
                time.sleep(float(attempt))
    raise RuntimeError(f"failed to initialize {name} after 3 attempts: {last_error}") from last_error


def _component_smoke(model: Any, name: str) -> dict[str, Any]:
    image = Image.new("RGB", (720, 160), "white")
    draw = ImageDraw.Draw(image)
    font = _font(52)
    text = "INVOICE TOTAL 123.45" if name != THAI_RECOGNIZER_MODEL else "ใบเสร็จ 123.45"
    draw.text((20, 40), text, fill="black", font=font)
    started = time.perf_counter()
    output = list(model.predict(input=np.asarray(image)))
    elapsed = time.perf_counter() - started
    return {
        "passed": bool(output),
        "result_count": len(output),
        "duration_seconds": elapsed,
        "input_size": list(image.size),
    }


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        Path("C:/Windows/Fonts/tahoma.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
    ):
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _file_manifest(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        records.append({
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return records


def _safe_remove_model(path: Path, official_root: Path, expected_name: str) -> None:
    resolved = path.resolve()
    root = official_root.resolve()
    if resolved.parent != root or resolved.name != expected_name:
        raise RuntimeError(f"refusing unsafe model removal: {resolved}")
    shutil.rmtree(resolved)


def _load_existing(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _package_exists(name: str) -> bool:
    try:
        importlib.metadata.version(name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
