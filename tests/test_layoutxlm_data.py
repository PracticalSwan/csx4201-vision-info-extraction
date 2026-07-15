from __future__ import annotations

import pytest

from src.information_extraction.layoutxlm_data import (
    LABEL_TO_ID,
    normalize_bbox,
    rotated_word_boxes,
    to_bio_labels,
)


def test_bio_labels_and_box_normalization() -> None:
    labels = to_bio_labels(["KEY", "KEY", "VALUE", "OTHER", "ANSWER", "ANSWER"])
    assert labels == ["B-KEY", "I-KEY", "B-VALUE", "O", "B-ANSWER", "I-ANSWER"]
    assert all(label in LABEL_TO_ID for label in labels)
    assert normalize_bbox([0, 0, 100, 200], 100, 200) == [0, 0, 1000, 1000]
    with pytest.raises(ValueError, match="outside"):
        normalize_bbox([-1, 0, 20, 20], 100, 200)


def test_rotated_word_boxes_are_valid_for_arbitrary_angle() -> None:
    example = {
        "width": 100,
        "height": 50,
        "tokens": [{
            "polygon": [[10, 10], [40, 10], [40, 20], [10, 20]],
            "bbox": [10, 10, 40, 20],
        }],
    }
    boxes, width, height, transform = rotated_word_boxes(example, 43)
    assert width > 100 and height > 50
    assert len(boxes) == 1
    assert all(0 <= value <= 1000 for value in boxes[0])
    assert boxes[0][0] < boxes[0][2]
    assert boxes[0][1] < boxes[0][3]
    assert transform["angle"] == 43
