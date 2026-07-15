"""Normalize PaddleOCR 3.x result objects into a stable token contract."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from src.information_extraction.geometry import bbox_to_polygon, polygon_to_bbox
from src.ocr.errors import OCRInferenceError
from src.rotation_common import canonical_json, stable_id


def normalize_paddle_result(
    results: Any,
    *,
    detector_model: str,
    recognizer_model: str,
    route: str,
    orientation: float,
    duration_seconds: float,
) -> dict[str, Any]:
    """Convert one-page PaddleOCR output across supported 3.x representations."""
    pages = list(results) if isinstance(results, (list, tuple)) else [results]
    if not pages:
        pages = [{}]
    payload = _as_mapping(pages[0])
    if "res" in payload and isinstance(payload["res"], Mapping):
        payload = dict(payload["res"])
    texts = _list(payload, "rec_texts", "texts", "text")
    scores = _list(payload, "rec_scores", "scores", "confidence")
    polygons = _list(payload, "rec_polys", "dt_polys", "polys", "polygons")
    boxes = _list(payload, "rec_boxes", "boxes")
    length = max(len(texts), len(scores), len(polygons), len(boxes), 0)
    words: list[dict[str, Any]] = []
    warnings: list[str] = []
    for index in range(length):
        text = str(texts[index]) if index < len(texts) else ""
        text = text.strip()
        if not text:
            continue
        score = _confidence(scores[index] if index < len(scores) else None)
        polygon_value = polygons[index] if index < len(polygons) else None
        box_value = boxes[index] if index < len(boxes) else None
        try:
            polygon = _polygon(polygon_value, box_value)
        except ValueError as exc:
            warnings.append(f"word {index} geometry skipped: {exc}")
            continue
        words.append({
            "id": stable_id("ocr", route, orientation, index, text),
            "text": text,
            "confidence": score,
            "polygon": polygon,
            "bbox": polygon_to_bbox(polygon),
        })
    lines = _lines_from_words(words)
    confidence_values = [word["confidence"] for word in words if word["confidence"] is not None]
    mean_confidence = float(np.mean(confidence_values)) if confidence_values else None
    normalized = {
        "full_text": "\n".join(line["text"] for line in lines),
        "words": words,
        "lines": lines,
        "mean_confidence": mean_confidence,
        "detector_model": detector_model,
        "recognizer_model": recognizer_model,
        "language_route": route,
        "orientation": float(orientation),
        "duration_seconds": max(0.0, float(duration_seconds)),
        "warnings": warnings,
    }
    normalized["provenance_hash"] = hashlib.sha256(canonical_json(normalized).encode("utf-8")).hexdigest()
    return normalized


def _as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    candidate = getattr(value, "json", None)
    if callable(candidate):
        candidate = candidate()
    if isinstance(candidate, str):
        try:
            candidate = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise OCRInferenceError("PaddleOCR result .json is invalid JSON") from exc
    if isinstance(candidate, Mapping):
        return dict(candidate)
    candidate = getattr(value, "res", None)
    if isinstance(candidate, Mapping):
        return dict(candidate)
    raise OCRInferenceError(f"unsupported PaddleOCR result type: {type(value).__name__}")


def _list(payload: Mapping[str, Any], *keys: str) -> list[Any]:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                return list(value)
            return [value]
    return []


def _confidence(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return max(0.0, min(1.0, result))


def _polygon(polygon: Any, box: Any) -> list[list[float]]:
    if polygon is not None:
        values = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
        if len(values) >= 4 and np.isfinite(values).all():
            return values.tolist()
    if box is not None:
        values = np.asarray(box, dtype=np.float64).reshape(-1)
        if len(values) == 4 and np.isfinite(values).all():
            return bbox_to_polygon(values.tolist())
    raise ValueError("missing finite polygon or box")


def _lines_from_words(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Paddle's rec polygons are line-level for the general OCR pipeline. Keep
    # one normalized line per returned recognition item instead of inventing
    # word segmentation that the recognizer did not provide.
    return [
        {
            "id": stable_id("line", word["id"]),
            "text": word["text"],
            "word_ids": [word["id"]],
            "polygon": word["polygon"],
            "bbox": word["bbox"],
            "confidence": word["confidence"],
        }
        for word in words
    ]
