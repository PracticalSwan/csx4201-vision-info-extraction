from __future__ import annotations

from pathlib import Path

import pytest

from src.inference.private_evaluation import (
    aggregate_private_results,
    anonymous_document_id,
    discover_private_documents,
    manual_review_rows,
)


def test_private_discovery_is_bounded_and_cannot_escape_root(tmp_path: Path) -> None:
    root = tmp_path / "private"
    root.mkdir()
    (root / "a.pdf").write_bytes(b"pdf")
    (root / "ignore.txt").write_text("x")
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"pdf")

    assert discover_private_documents(root, explicit_files=["a.pdf"], limit=1) == [root / "a.pdf"]
    with pytest.raises(ValueError, match="within"):
        discover_private_documents(root, explicit_files=[str(outside)])


def test_private_rows_use_only_anonymous_identity() -> None:
    result = {
        "document_type": {"label": "invoice"},
        "fields": {
            "total_amount": {"value": "12.50", "confidence": 0.8, "page_number": 2},
            "date": None,
        },
    }
    rows = manual_review_rows(anonymous_document_id(1), result)
    assert rows == [{
        "anonymous_document_id": "private_000001",
        "predicted_document_type": "invoice",
        "extracted_field": "total_amount",
        "predicted_value": "12.50",
        "confidence": 0.8,
        "evidence_page": 2,
        "user_corrected_value": "",
        "correct_yes_no": "",
        "notes": "",
    }]


def test_private_aggregate_contains_no_row_level_payload() -> None:
    result = {
        "document_type": {"label": "invoice"},
        "fields": {"total_amount": {"value": "12.50"}},
        "pages": [{
            "ocr": {"words": [{"text": "secret"}], "language_route": "general", "mean_confidence": 0.9},
            "entities": [{}], "key_value_pairs": [{}],
        }],
        "processing": {"duration_seconds": 1.0},
    }
    report = aggregate_private_results(
        [result], attempted_documents=1, error_type_counts={}, elapsed_seconds=2.0,
        checkpoint_model_sha256="abc",
    )
    serialized = str(report)
    assert "secret" not in serialized
    assert report["contains_ocr_text"] is False
    assert report["gmail_fit_rows"] == 0
