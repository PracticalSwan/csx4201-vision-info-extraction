"""Deterministic calibration metrics and abstention-threshold selection."""
from __future__ import annotations

import math
from collections.abc import Sequence


def apply_abstention(
    confidences: Sequence[float],
    predictions: Sequence[int],
    *,
    threshold: float,
    class_names: Sequence[str] | None = None,
    per_class_thresholds: dict[str, float] | None = None,
) -> list[int]:
    """Map predictions below their calibrated threshold to the null class."""
    if len(confidences) != len(predictions):
        raise ValueError("confidence and prediction lengths differ")
    thresholds = dict(per_class_thresholds or {})
    retained = []
    for confidence, prediction in zip(confidences, predictions, strict=True):
        prediction_id = int(prediction)
        selected_threshold = float(threshold)
        if class_names is not None and 0 <= prediction_id < len(class_names):
            selected_threshold = float(
                thresholds.get(class_names[prediction_id], thresholds.get("default", threshold))
            )
        retained.append(
            prediction_id if float(confidence) >= selected_threshold else 0
        )
    return retained


def positive_f1_at_threshold(
    confidences: Sequence[float],
    predictions: Sequence[int],
    labels: Sequence[int],
    *,
    threshold: float,
) -> float:
    if not (len(confidences) == len(predictions) == len(labels)):
        raise ValueError("confidence, prediction, and label lengths differ")
    tp = fp = fn = 0
    for confidence, predicted, actual in zip(
        confidences, predictions, labels, strict=True
    ):
        retained_prediction = int(predicted) if float(confidence) >= threshold else 0
        actual_positive = int(actual) > 0
        predicted_positive = retained_prediction > 0
        if actual_positive and retained_prediction == int(actual):
            tp += 1
        else:
            fp += int(predicted_positive)
            fn += int(actual_positive)
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def choose_positive_threshold(
    confidences: Sequence[float],
    predictions: Sequence[int],
    labels: Sequence[int],
) -> float:
    """Maximize positive-class micro F1; break ties toward more abstention."""
    if not confidences:
        return 0.50
    actual_positive_count = sum(int(label) > 0 for label in labels)
    ranked_predictions = sorted(
        (
            max(0.0, min(1.0, float(confidence))),
            int(predicted),
            int(actual),
        )
        for confidence, predicted, actual in zip(
            confidences, predictions, labels, strict=True
        )
        if int(predicted) > 0
    )
    ranked_predictions.reverse()
    best_f1, best_threshold = 0.0, 1.0
    tp = fp = 0
    index = 0
    while index < len(ranked_predictions):
        threshold = ranked_predictions[index][0]
        while index < len(ranked_predictions) and ranked_predictions[index][0] == threshold:
            _, predicted, actual = ranked_predictions[index]
            if actual > 0 and predicted == actual:
                tp += 1
            else:
                fp += 1
            index += 1
        fn = actual_positive_count - tp
        denominator = 2 * tp + fp + fn
        score = 2 * tp / denominator if denominator else 0.0
        if (score, threshold) > (best_f1, best_threshold):
            best_f1, best_threshold = score, threshold
    return best_threshold


def choose_document_threshold(
    confidences: Sequence[float],
    correct: Sequence[bool],
    *,
    minimum_coverage: float = 0.90,
) -> float:
    """Maximize selective accuracy while retaining the requested coverage."""
    if len(confidences) != len(correct):
        raise ValueError("document confidence and correctness lengths differ")
    if not confidences:
        return 0.50
    if not 0.0 < minimum_coverage <= 1.0:
        raise ValueError("minimum coverage must be in (0, 1]")
    ranked_values = sorted(
        (
            max(0.0, min(1.0, float(confidence))),
            bool(value),
        )
        for confidence, value in zip(confidences, correct, strict=True)
    )
    ranked_values.reverse()
    best: tuple[float, float, float] | None = None
    retained_count = correct_count = 0
    index = 0
    while index < len(ranked_values):
        threshold = ranked_values[index][0]
        while index < len(ranked_values) and ranked_values[index][0] == threshold:
            correct_count += int(ranked_values[index][1])
            retained_count += 1
            index += 1
        coverage = retained_count / len(correct)
        if coverage + 1e-12 < minimum_coverage:
            continue
        candidate = (correct_count / retained_count, threshold, coverage)
        if best is None or candidate > best:
            best = candidate
    if best is None:
        return 0.0
    return best[1]


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool],
    *,
    bins: int = 15,
) -> float:
    if len(confidences) != len(correct):
        raise ValueError("confidence and correctness lengths differ")
    if bins < 1:
        raise ValueError("bins must be positive")
    if not confidences:
        return 0.0
    grouped: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for confidence, value in zip(confidences, correct, strict=True):
        bounded = max(0.0, min(1.0, float(confidence)))
        index = min(bins - 1, int(math.floor(bounded * bins)))
        grouped[index].append((bounded, bool(value)))
    total = len(confidences)
    return sum(
        len(values) / total
        * abs(
            sum(confidence for confidence, _ in values) / len(values)
            - sum(value for _, value in values) / len(values)
        )
        for values in grouped
        if values
    )
