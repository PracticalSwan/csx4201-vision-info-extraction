from __future__ import annotations

from src.information_extraction.alignment import align_ocr_to_annotations, normalize_alignment_text


def _token(text: str, bbox: list[int], label: str = "OTHER") -> dict:
    return {"text": text, "bbox": bbox, "entity_label": label}


def test_alignment_handles_exact_merged_split_unicode_and_unmatched() -> None:
    annotations = [
        _token("Invoice", [0, 0, 40, 10], "KEY"),
        _token("No.", [45, 0, 60, 10], "KEY"),
        _token("İstanbul", [0, 20, 70, 30], "VALUE"),
        _token("ยอดรวม", [0, 40, 60, 50], "KEY"),
        _token("missing", [0, 60, 50, 70], "VALUE"),
    ]
    ocr = [
        _token("Invoice No.", [0, 0, 60, 10]),
        _token("Istanbul", [0, 20, 70, 30]),
        _token("ยอด", [0, 40, 30, 50]),
        _token("รวม", [31, 40, 60, 50]),
        _token("extra", [0, 80, 40, 90]),
    ]
    result = align_ocr_to_annotations(ocr, annotations)
    assert result["merged_matches"] >= 1
    assert result["split_matches"] >= 1
    assert result["alignment_coverage"] >= 0.8
    assert result["ocr_labels"][0] == "KEY"
    assert result["ocr_labels"][-1] == "OTHER"
    assert len(result["unmatched_labels"]) == 1
    assert normalize_alignment_text("İSTANBUL") == normalize_alignment_text("istanbul")
