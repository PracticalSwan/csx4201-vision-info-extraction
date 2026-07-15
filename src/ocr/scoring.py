"""Auditable OCR candidate quality scoring with route calibration."""
from __future__ import annotations

import math
import statistics
import unicodedata
from collections import Counter
from collections.abc import Mapping
from typing import Any

import numpy as np

DEFAULT_ROUTE_CALIBRATION = {
    "general": {"confidence_center": 0.75, "confidence_scale": 0.20},
    "thai": {"confidence_center": 0.72, "confidence_scale": 0.22},
}


def score_ocr_candidate(
    result: Mapping[str, Any], image_width: int, image_height: int,
    calibration: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, float]:
    words = list(result.get("words") or [])
    texts = [str(word.get("text", "")) for word in words if str(word.get("text", "")).strip()]
    confidences = [
        float(word["confidence"]) for word in words
        if word.get("confidence") is not None and math.isfinite(float(word["confidence"]))
    ]
    route = str(result.get("language_route", "general"))
    calibration_values = dict((calibration or DEFAULT_ROUTE_CALIBRATION).get(route, {}))
    center = calibration_values.get("confidence_center", 0.75)
    scale = max(1e-6, calibration_values.get("confidence_scale", 0.2))
    mean_confidence = statistics.fmean(confidences) if confidences else 0.0
    median_confidence = statistics.median(confidences) if confidences else 0.0
    calibrated_confidence = 1.0 / (1.0 + math.exp(-(mean_confidence - center) / scale))
    printable = sum(character.isprintable() and not unicodedata.category(character).startswith("C") for text in texts for character in text)
    characters = max(1, sum(len(text) for text in texts))
    valid_character_ratio = printable / characters
    thai_chars = sum("\u0e00" <= character <= "\u0e7f" for text in texts for character in text)
    script_chars = sum(
        character.isalpha() or "\u0e00" <= character <= "\u0e7f"
        for text in texts
        for character in text
    )
    thai_ratio = thai_chars / max(1, script_chars)
    script_consistency = thai_ratio if route == "thai" else 1.0 - thai_ratio
    coverage = min(1.0, sum(_box_area(word.get("bbox")) for word in words) / max(1.0, image_width * image_height))
    word_count_score = min(1.0, math.log1p(len(texts)) / math.log(41.0))
    empty_line_rate = 0.0 if texts else 1.0
    duplicates = Counter(text.casefold().strip() for text in texts)
    duplicate_count = sum(max(0, count - 1) for count in duplicates.values())
    duplicate_penalty = duplicate_count / max(1, len(texts))
    garbage_penalty = _garbage_ratio(texts)
    alignment = _line_alignment(words)
    low_outlier_penalty = (
        sum(value < 0.25 for value in confidences) / len(confidences) if confidences else 1.0
    )
    total = (
        0.22 * word_count_score
        + 0.22 * calibrated_confidence
        + 0.08 * median_confidence
        + 0.10 * min(1.0, coverage * 8.0)
        + 0.13 * valid_character_ratio
        + 0.10 * script_consistency
        + 0.10 * alignment
        - 0.025 * duplicate_penalty
        - 0.025 * garbage_penalty
        - 0.025 * empty_line_rate
        - 0.025 * low_outlier_penalty
    )
    return {
        "total": float(max(0.0, min(1.0, total))),
        "word_count": float(len(texts)),
        "word_count_score": float(word_count_score),
        "mean_confidence": float(mean_confidence),
        "median_confidence": float(median_confidence),
        "route_calibrated_confidence": float(calibrated_confidence),
        "text_detection_coverage": float(coverage),
        "valid_character_ratio": float(valid_character_ratio),
        "thai_script_ratio": float(thai_ratio),
        "script_consistency": float(script_consistency),
        "line_alignment": float(alignment),
        "duplicate_penalty": float(duplicate_penalty),
        "garbage_penalty": float(garbage_penalty),
        "empty_line_rate": float(empty_line_rate),
        "outlier_confidence_penalty": float(low_outlier_penalty),
    }


def _box_area(value: Any) -> float:
    try:
        x0, y0, x1, y1 = map(float, value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _garbage_ratio(texts: list[str]) -> float:
    if not texts:
        return 1.0
    bad = 0
    for text in texts:
        alnum = sum(character.isalnum() for character in text)
        repeated = max(Counter(text).values(), default=0) / max(1, len(text))
        if alnum == 0 or repeated > 0.8:
            bad += 1
    return bad / len(texts)


def _line_alignment(words: list[Mapping[str, Any]]) -> float:
    angles = []
    for word in words:
        polygon = np.asarray(word.get("polygon") or [], dtype=np.float64)
        if polygon.shape[0] < 2 or polygon.shape[1:] != (2,):
            continue
        delta = polygon[1] - polygon[0]
        angles.append(abs(math.atan2(float(delta[1]), float(delta[0]))))
    if not angles:
        return 0.0
    return float(max(0.0, 1.0 - statistics.median(angles) / (math.pi / 2.0)))
