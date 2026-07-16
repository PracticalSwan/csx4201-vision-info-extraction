from __future__ import annotations

import pytest

from src.information_extraction.multitask_calibration import (
    apply_abstention,
    choose_document_threshold,
    choose_positive_threshold,
    expected_calibration_error,
    positive_f1_at_threshold,
)


def test_apply_abstention_supports_per_class_thresholds() -> None:
    assert apply_abstention(
        [0.80, 0.80, 0.40],
        [1, 2, 1],
        threshold=0.50,
        class_names=["O", "date", "total"],
        per_class_thresholds={"date": 0.75, "total": 0.90},
    ) == [1, 0, 0]


def test_positive_threshold_improves_f1_and_prefers_conservative_tie() -> None:
    confidences = [0.90, 0.80, 0.40, 0.90]
    predictions = [1, 1, 1, 0]
    labels = [1, 1, 0, 0]

    threshold = choose_positive_threshold(confidences, predictions, labels)

    assert threshold == pytest.approx(0.80)
    assert positive_f1_at_threshold(
        confidences, predictions, labels, threshold=threshold
    ) == pytest.approx(1.0)


def test_document_threshold_preserves_coverage_and_rejects_low_confidence_error() -> None:
    confidences = [0.95] * 9 + [0.30]
    correct = [True] * 9 + [False]

    threshold = choose_document_threshold(
        confidences,
        correct,
        minimum_coverage=0.90,
    )

    assert 0.30 < threshold <= 0.95
    retained = [ok for confidence, ok in zip(confidences, correct) if confidence >= threshold]
    assert len(retained) / len(correct) >= 0.90
    assert all(retained)


def test_expected_calibration_error_is_bounded() -> None:
    value = expected_calibration_error(
        [0.95, 0.80, 0.20, 0.10],
        [True, True, False, False],
        bins=4,
    )
    assert 0.0 <= value <= 1.0
    assert value == pytest.approx(0.1375)
