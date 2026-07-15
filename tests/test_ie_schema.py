from __future__ import annotations

import copy

import pytest

from src.information_extraction.schema import (
    CANONICAL_FIELDS,
    OutputValidationError,
    build_document_result,
    validate_document_result,
)


def _page() -> dict:
    identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    return {
        "page_number": 1,
        "width": 100,
        "height": 200,
        "selected_ocr_orientation": 0,
        "full_text": "",
        "ocr": {
            "detector_model": "PP-OCRv6_medium_det",
            "recognizer_model": "PP-OCRv6_medium_rec",
            "language_route": "general",
            "mean_confidence": None,
            "words": [],
            "lines": [],
            "candidate_scores": [],
            "provenance_hash": "0123456789abcdef",
            "duration_seconds": 0.0,
        },
        "entities": [],
        "key_value_pairs": [],
        "tables": [],
        "warnings": [],
        "transforms": {"forward": identity, "inverse": identity},
    }


def test_empty_result_has_all_null_canonical_fields() -> None:
    result = build_document_result(
        document_id="doc_1",
        source_type="image",
        pages=[_page()],
        device="cpu",
    )
    assert tuple(result["fields"]) == CANONICAL_FIELDS
    assert all(value is None for value in result["fields"].values())
    assert result["rotation_display"]["purpose"] == "display_only"


def test_field_requires_evidence() -> None:
    result = build_document_result(
        document_id="doc_1",
        source_type="image",
        pages=[_page()],
        device="cpu",
    )
    invalid = copy.deepcopy(result)
    invalid["fields"]["total_amount"] = "12.50"
    with pytest.raises(OutputValidationError, match="total_amount"):
        validate_document_result(invalid)


def test_unknown_canonical_field_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported canonical fields"):
        build_document_result(
            document_id="doc_1",
            source_type="image",
            pages=[_page()],
            device="cpu",
            fields={"invented": None},
        )
