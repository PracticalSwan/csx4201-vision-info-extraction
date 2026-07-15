"""Dependency-light OCR, entity, relation, and field metrics."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import cv2
import numpy as np


def normalized_text(value: Any) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value)).casefold()).strip()


def edit_distance(left: Sequence[Any], right: Sequence[Any]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(min(
                current[-1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1] + (left_value != right_value),
            ))
        previous = current
    return previous[-1]


def ocr_text_metrics(reference: str, prediction: str) -> dict[str, float | int]:
    reference_normalized = normalized_text(reference)
    prediction_normalized = normalized_text(prediction)
    reference_words = reference_normalized.split()
    prediction_words = prediction_normalized.split()
    character_errors = edit_distance(reference_normalized, prediction_normalized)
    word_errors = edit_distance(reference_words, prediction_words)
    return {
        "reference_characters": len(reference_normalized),
        "prediction_characters": len(prediction_normalized),
        "character_errors": character_errors,
        "word_errors": word_errors,
        "cer": character_errors / max(1, len(reference_normalized)),
        "wer": word_errors / max(1, len(reference_words)),
        "recognized_text_coverage": max(
            0.0, 1.0 - character_errors / max(1, len(reference_normalized))
        ),
        "empty_output": int(not prediction_normalized),
    }


def text_detection_metrics(
    reference_items: Sequence[Mapping[str, Any]],
    predicted_items: Sequence[Mapping[str, Any]],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Match convex text polygons one-to-one and report detection P/R/F1."""
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")
    references = [_polygon(item) for item in reference_items]
    predictions = [_polygon(item) for item in predicted_items]
    references = [value for value in references if value is not None]
    predictions = [value for value in predictions if value is not None]
    candidates: list[tuple[float, int, int]] = []
    for reference_index, reference in enumerate(references):
        for prediction_index, prediction in enumerate(predictions):
            iou = _convex_polygon_iou(reference, prediction)
            if iou >= iou_threshold:
                candidates.append((iou, reference_index, prediction_index))
    matched_references: set[int] = set()
    matched_predictions: set[int] = set()
    matched_ious: list[float] = []
    for iou, reference_index, prediction_index in sorted(candidates, reverse=True):
        if reference_index in matched_references or prediction_index in matched_predictions:
            continue
        matched_references.add(reference_index)
        matched_predictions.add(prediction_index)
        matched_ious.append(iou)
    true_positive = len(matched_ious)
    expected = len(references)
    predicted = len(predictions)
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / expected if expected else None
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if recall is not None and precision + recall else 0.0 if recall is not None else None
    )
    return {
        "reference_available": bool(expected),
        "true_positive": true_positive,
        "expected": expected,
        "predicted": predicted,
        "precision": precision if expected else None,
        "recall": recall,
        "f1": f1,
        "mean_matched_iou": sum(matched_ious) / len(matched_ious) if matched_ious else None,
        "iou_threshold": float(iou_threshold),
    }


def extraction_metrics(
    annotation: Mapping[str, Any], prediction_page: Mapping[str, Any], fields: Mapping[str, Any]
) -> dict[str, Any]:
    entity = _set_metrics(
        Counter(_entity_key(value) for value in annotation.get("entities") or []),
        Counter(_entity_key(value) for value in prediction_page.get("entities") or []),
    )
    relation = _set_metrics(
        _relation_counter(annotation.get("entities") or [], annotation.get("relations") or []),
        _relation_counter(prediction_page.get("entities") or [], prediction_page.get("key_value_pairs") or []),
    )
    expected_fields = annotation.get("canonical_fields") or {}
    applicable = 0
    correct = 0
    for field, expected in expected_fields.items():
        if not isinstance(expected, Mapping) or expected.get("value") in (None, ""):
            continue
        applicable += 1
        predicted = fields.get(field)
        if isinstance(predicted, Mapping) and normalized_text(predicted.get("value")) == normalized_text(expected["value"]):
            correct += 1
    return {
        "entity": entity,
        "relation": relation,
        "canonical_fields": {
            "applicable": applicable,
            "correct": correct,
            "accuracy": correct / applicable if applicable else None,
        },
    }


def _entity_key(entity: Mapping[str, Any]) -> tuple[str, str]:
    return str(entity.get("label", "")), normalized_text(entity.get("text", ""))


def _relation_counter(
    entities: Sequence[Mapping[str, Any]], relations: Sequence[Mapping[str, Any]]
) -> Counter[tuple[str, tuple[str, str], tuple[str, str]]]:
    index = {str(entity.get("id")): _entity_key(entity) for entity in entities}
    result: Counter[tuple[str, tuple[str, str], tuple[str, str]]] = Counter()
    for relation in relations:
        source = index.get(str(relation.get("source_id")))
        target = index.get(str(relation.get("target_id")))
        if source and target:
            result[(str(relation.get("type", "GENERIC")), source, target)] += 1
    return result


def _set_metrics(expected: Counter[Any], predicted: Counter[Any]) -> dict[str, float | int]:
    true_positive = sum((expected & predicted).values())
    expected_count = sum(expected.values())
    predicted_count = sum(predicted.values())
    precision = true_positive / predicted_count if predicted_count else 0.0
    recall = true_positive / expected_count if expected_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": true_positive,
        "expected": expected_count,
        "predicted": predicted_count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _polygon(item: Mapping[str, Any]) -> np.ndarray | None:
    value = item.get("polygon")
    if not value and item.get("bbox"):
        x0, y0, x1, y1 = map(float, item["bbox"])
        value = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    try:
        points = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
        return None
    hull = cv2.convexHull(points).reshape(-1, 2)
    return hull if len(hull) >= 3 and cv2.contourArea(hull) > 0 else None


def _convex_polygon_iou(left: np.ndarray, right: np.ndarray) -> float:
    left_area = float(cv2.contourArea(left))
    right_area = float(cv2.contourArea(right))
    intersection_area, _ = cv2.intersectConvexConvex(left, right)
    union = left_area + right_area - float(intersection_area)
    return max(0.0, min(1.0, float(intersection_area) / union)) if union > 0 else 0.0
