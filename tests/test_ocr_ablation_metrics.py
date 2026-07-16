from __future__ import annotations

from scripts.run_ocr_preprocessing_ablation import _aggregate_setting, word_error_rate


def test_word_error_rate_handles_insert_delete_and_substitute() -> None:
    assert word_error_rate(["a", "b"], ["a", "b"]) == 0.0
    assert word_error_rate(["a", "b"], ["a"]) == 0.5
    assert word_error_rate(["a"], ["b", "c"]) == 2.0
    assert word_error_rate([], []) == 0.0


def test_successful_ablation_setting_is_explicitly_selectable() -> None:
    report = _aggregate_setting(
        "original",
        "original",
        {},
        [{
            "dataset": "funsd",
            "alignment_coverage": 0.75,
            "word_error_rate": 0.25,
            "mean_confidence": 0.9,
            "word_count": 12,
            "selected_orientation": 0.0,
        }],
    )

    assert report["status"] == "passed"
    assert report["mean_alignment_coverage"] == 0.75
