from __future__ import annotations

import pytest

from src.information_extraction.multitask_data import (
    CANONICAL_FIELD_LABELS,
    CANONICAL_FIELD_TO_ID,
    RELATION_LABELS,
)
from src.information_extraction.multitask_inference import (
    _load_calibration,
    apply_confidence_floor,
    aggregate_word_probabilities,
    decode_canonical_fields,
    decode_document_type,
    decode_relations,
    reconstruct_ocr_tables,
    reconstruct_tables,
)


def test_explicit_missing_calibration_fails_closed(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    missing = tmp_path / "missing-calibration.json"

    with pytest.raises(FileNotFoundError, match="required calibration file"):
        _load_calibration(missing, checkpoint)


def test_confidence_floor_never_weakens_fitted_thresholds() -> None:
    calibrated = {
        "temperatures": {"entity": 1.0},
        "thresholds": {
            "entity": 0.7,
            "canonical": 0.4,
            "relation": 0.2,
            "document": 0.9,
        },
        "canonical_field_thresholds": {"total_amount": 0.8, "date": 0.3},
    }

    result = apply_confidence_floor(calibrated, 0.6)

    assert result["thresholds"] == {
        "entity": 0.7,
        "canonical": 0.6,
        "relation": 0.6,
        "document": 0.9,
    }
    assert result["canonical_field_thresholds"] == {
        "total_amount": 0.8,
        "date": 0.6,
    }
    assert calibrated["thresholds"]["canonical"] == 0.4


def _distribution(size: int, label: int, confidence: float) -> list[float]:
    remainder = (1.0 - confidence) / (size - 1)
    values = [remainder] * size
    values[label] = confidence
    return values


def test_overlapping_windows_average_one_subtoken_per_word() -> None:
    probabilities = [
        [
            [0.5, 0.5],
            [0.2, 0.8],
            [0.9, 0.1],
            [0.6, 0.4],
            [0.5, 0.5],
        ],
        [
            [0.5, 0.5],
            [0.2, 0.8],
            [0.1, 0.9],
            [0.5, 0.5],
        ],
    ]
    word_ids = [[None, 0, 0, 1, None], [None, 1, 2, None]]

    result = aggregate_word_probabilities(probabilities, word_ids, word_count=3)

    assert result[0] == pytest.approx([0.2, 0.8])
    assert result[1] == pytest.approx([0.4, 0.6])
    assert result[2] == pytest.approx([0.1, 0.9])


def test_canonical_evidence_decoding_is_thresholded_and_evidence_bearing() -> None:
    words = [
        {
            "id": "w1",
            "text": "123.45",
            "bbox": [10, 20, 60, 35],
            "polygon": [[10, 20], [60, 20], [60, 35], [10, 35]],
        },
        {
            "id": "w2",
            "text": "USD",
            "bbox": [65, 20, 95, 35],
            "polygon": [[65, 20], [95, 20], [95, 35], [65, 35]],
        },
    ]
    probabilities = [
        _distribution(
            len(CANONICAL_FIELD_LABELS),
            CANONICAL_FIELD_TO_ID["total_amount"],
            0.90,
        ),
        _distribution(
            len(CANONICAL_FIELD_LABELS),
            CANONICAL_FIELD_TO_ID["currency"],
            0.85,
        ),
    ]

    fields = decode_canonical_fields(
        words,
        probabilities,
        page_number=2,
        thresholds={"total_amount": 0.80, "currency": 0.90},
    )

    assert fields["total_amount"]["value"] == "123.45"
    assert fields["total_amount"]["raw_text"] == "123.45"
    assert fields["total_amount"]["page_number"] == 2
    assert fields["total_amount"]["method"] == "model:canonical_evidence"
    assert "currency" not in fields


def test_document_and_relation_decoding_abstain_below_threshold() -> None:
    label, confidence = decode_document_type(
        [0.1, 0.7, 0.1, 0.1], threshold=0.6
    )
    assert label == "receipt"
    assert confidence == pytest.approx(0.7)
    assert decode_document_type([0.3, 0.3, 0.2, 0.2], threshold=0.6)[0] == "unknown"

    pairs = [{
        "source_id": "key",
        "target_id": "value",
        "candidate_relation_type": "KEY_VALUE",
    }]
    relation_id = RELATION_LABELS.index("KEY_VALUE")
    relations = decode_relations(
        pairs,
        [_distribution(len(RELATION_LABELS), relation_id, 0.8)],
        page_number=1,
        threshold=0.7,
    )
    assert [(item["source_id"], item["target_id"], item["type"]) for item in relations] == [
        ("key", "value", "KEY_VALUE")
    ]
    assert decode_relations(
        pairs,
        [_distribution(len(RELATION_LABELS), relation_id, 0.6)],
        page_number=1,
        threshold=0.7,
    ) == []


def test_table_reconstruction_is_geometry_based_and_explicit() -> None:
    entities = []
    for index, (text, bbox) in enumerate((
        ("A", [0, 0, 40, 10]),
        ("B", [50, 0, 90, 10]),
        ("1", [0, 20, 40, 30]),
        ("2", [50, 20, 90, 30]),
    )):
        entities.append({
            "id": f"cell-{index}",
            "label": "TABLE_CELL",
            "text": text,
            "bbox": bbox,
            "confidence": 0.9,
            "page_number": 1,
        })

    tables = reconstruct_tables(entities, page_number=1)

    assert len(tables) == 1
    assert tables[0]["method"] == "geometry:table_cell_grid"
    assert tables[0]["row_count"] == 2
    assert tables[0]["column_count"] == 2
    assert tables[0]["source"] == "model_entities"
    assert tables[0]["rows"][0]["row_type"] == "unknown"
    assert tables[0]["raw_ocr_fallback"] == "A | B\n1 | 2"
    assert [(cell["row_index"], cell["column_index"], cell["text"]) for cell in tables[0]["cells"]] == [
        (0, 0, "A"),
        (0, 1, "B"),
        (1, 0, "1"),
        (1, 1, "2"),
    ]


def test_ocr_table_fallback_requires_repeated_aligned_rows_and_labels_headers() -> None:
    words = []
    for row, values in enumerate((
        ("Description", "Qty", "Total"),
        ("Coffee", "2", "10.00"),
        ("Tea", "1", "4.00"),
    )):
        for column, text in enumerate(values):
            x = column * 100
            y = row * 25
            words.append({
                "id": f"w-{row}-{column}", "text": text,
                "bbox": [x, y, x + 60, y + 12], "confidence": 0.9,
            })

    tables = reconstruct_ocr_tables(words, page_number=1)

    assert len(tables) == 1
    assert tables[0]["source"] == "ocr_geometry"
    assert tables[0]["header_row_index"] == 0
    assert [cell["semantic_type"] for cell in tables[0]["rows"][1]["cells"]] == [
        "description", "quantity", "line_total",
    ]
    assert all(row["row_type"] == "item" for row in tables[0]["rows"][1:])
