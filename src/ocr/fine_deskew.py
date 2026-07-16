"""Residual text-angle estimation from OCR polygon baselines."""
from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any


def estimate_residual_deskew(
    words: Sequence[Mapping[str, Any]],
    *,
    minimum_edge_length: float = 8.0,
    maximum_absolute_angle: float = 45.0,
) -> dict[str, float | int]:
    """Estimate a bounded corrective angle without any K-Means dependency."""
    observations: list[tuple[float, float]] = []
    confidences = []
    baseline_lengths = []
    for word in words:
        polygon = word.get("polygon") or []
        if len(polygon) < 4:
            continue
        edges = ((polygon[0], polygon[1]), (polygon[3], polygon[2]))
        candidates = []
        for left, right in edges:
            try:
                dx = float(right[0]) - float(left[0])
                dy = float(right[1]) - float(left[1])
            except (TypeError, ValueError, IndexError):
                continue
            length = math.hypot(dx, dy)
            if length < minimum_edge_length:
                continue
            if dx < 0:
                dx, dy = -dx, -dy
            angle = math.degrees(math.atan2(dy, dx))
            while angle >= 90.0:
                angle -= 180.0
            while angle < -90.0:
                angle += 180.0
            if abs(angle) <= maximum_absolute_angle:
                candidates.append((length, angle))
        if not candidates:
            continue
        length, angle = max(candidates)
        confidence = float(word.get("confidence", 1.0) or 0.0)
        weight = length * max(0.1, min(1.0, confidence))
        observations.append((angle, weight))
        confidences.append(max(0.0, min(1.0, confidence)))
        baseline_lengths.append(length)
    if not observations:
        return {
            "text_angle_degrees": 0.0,
            "correction_degrees": 0.0,
            "reliability": 0.0,
            "dispersion_degrees": 90.0,
            "supported_word_count": 0,
        }
    angle = _weighted_median(observations)
    dispersion = _weighted_median(
        [(abs(value - angle), weight) for value, weight in observations]
    )
    # A single long recognized line is stronger evidence than several tiny
    # words. Accept either broad word support or at least ~300 px of baseline.
    support = max(
        min(1.0, len(observations) / 6.0),
        min(1.0, sum(baseline_lengths) / 300.0),
    )
    concentration = max(0.0, 1.0 - dispersion / 12.0)
    confidence_factor = statistics.fmean(confidences) if confidences else 0.0
    reliability = support * concentration * confidence_factor
    return {
        "text_angle_degrees": float(angle),
            # Polygon angles are measured in downward-positive image
            # coordinates. The visual-counterclockwise correction used by the
            # geometry module has the same signed value.
            "correction_degrees": float(angle),
        "reliability": float(max(0.0, min(1.0, reliability))),
        "dispersion_degrees": float(dispersion),
        "supported_word_count": len(observations),
    }


def _weighted_median(values: Sequence[tuple[float, float]]) -> float:
    ordered = sorted(values)
    total = sum(max(0.0, weight) for _, weight in ordered)
    threshold = total / 2.0
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += max(0.0, weight)
        if cumulative >= threshold:
            return float(value)
    return float(ordered[-1][0])
