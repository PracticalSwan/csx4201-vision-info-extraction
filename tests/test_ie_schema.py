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


def test_table_output_has_a_schema_validated_geometry_contract() -> None:
    page = _page()
    page["tables"] = [{
        "id": "table-1",
        "page_number": 1,
        "method": "geometry:table_cell_grid",
        "source": "model_entities",
        "confidence": 0.9,
        "row_count": 1,
        "column_count": 1,
        "bbox": [0, 0, 20, 10],
        "cells": [{
            "row_index": 0,
            "column_index": 0,
            "text": "A",
            "bbox": [0, 0, 20, 10],
            "polygon": [[0, 0], [20, 0], [20, 10], [0, 10]],
            "entity_id": "entity-1",
            "confidence": 0.9,
            "semantic_type": "unknown",
        }],
        "header_row_index": None,
        "rows": [{
            "row_index": 0,
            "row_type": "unknown",
            "cells": [{
                "row_index": 0, "column_index": 0, "text": "A",
                "bbox": [0, 0, 20, 10],
                "polygon": [[0, 0], [20, 0], [20, 10], [0, 10]],
                "entity_id": "entity-1", "confidence": 0.9,
                "semantic_type": "unknown",
            }],
        }],
        "source_polygons": [[[0, 0], [20, 0], [20, 10], [0, 10]]],
        "warnings": [],
        "raw_ocr_fallback": "A",
    }]
    result = build_document_result(
        document_id="doc_1",
        source_type="image",
        pages=[page],
        device="cpu",
    )
    validate_document_result(result)

    invalid = copy.deepcopy(result)
    del invalid["pages"][0]["tables"][0]["row_count"]
    with pytest.raises(OutputValidationError, match="row_count"):
        validate_document_result(invalid)
