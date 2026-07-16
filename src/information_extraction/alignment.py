"""Text-and-geometry OCR alignment with explicit unmatched accounting."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from difflib import SequenceMatcher
from typing import Any


def align_ocr_to_annotations(
    ocr_tokens: Sequence[Mapping[str, Any]],
    annotation_tokens: Sequence[Mapping[str, Any]],
    *,
    minimum_score: float = 0.45,
) -> dict[str, Any]:
    """Greedily align OCR items to source tokens without dropping failures."""
    matches: list[dict[str, Any]] = []
    annotation_to_ocr: dict[int, list[int]] = {}
    ocr_to_annotation: dict[int, list[int]] = {}
    candidates: list[tuple[float, int, int, str]] = []
    for annotation_index, annotation in enumerate(annotation_tokens):
        source_text = normalize_alignment_text(str(annotation.get("text", "")))
        if not source_text:
            continue
        for ocr_index, ocr in enumerate(ocr_tokens):
            ocr_text = normalize_alignment_text(str(ocr.get("text", "")))
            if not ocr_text:
                continue
            text_score, match_type = _text_score(source_text, ocr_text)
            geometry_score = _geometry_score(annotation.get("bbox"), ocr.get("bbox"))
            score = 0.7 * text_score + 0.3 * geometry_score
            if score >= minimum_score:
                candidates.append((score, annotation_index, ocr_index, match_type))
    # Prefer high-quality matches. Allow one-to-many/many-to-one only for
    # containment matches, which cover line-level OCR versus word annotations.
    for score, annotation_index, ocr_index, match_type in sorted(candidates, reverse=True):
        existing_annotation = annotation_to_ocr.get(annotation_index, [])
        existing_ocr = ocr_to_annotation.get(ocr_index, [])
        if existing_annotation and match_type not in {"annotation_contains_ocr", "ocr_contains_annotation"}:
            continue
        if existing_ocr and match_type not in {"annotation_contains_ocr", "ocr_contains_annotation"}:
            continue
        if ocr_index in existing_annotation or annotation_index in existing_ocr:
            continue
        annotation_to_ocr.setdefault(annotation_index, []).append(ocr_index)
        ocr_to_annotation.setdefault(ocr_index, []).append(annotation_index)
        matches.append({
            "annotation_index": annotation_index,
            "ocr_index": ocr_index,
            "score": score,
            "match_type": match_type,
        })
    labels: list[str] = []
    for ocr_index in range(len(ocr_tokens)):
        annotation_indices = ocr_to_annotation.get(ocr_index, [])
        if not annotation_indices:
            labels.append("OTHER")
            continue
        source_labels = [
            str(annotation_tokens[index].get("entity_label", "OTHER"))
            for index in annotation_indices
        ]
        labels.append(_resolve_label(source_labels))
    matched_annotations = set(annotation_to_ocr)
    matched_ocr = set(ocr_to_annotation)
    split_matches = sum(len(values) > 1 for values in annotation_to_ocr.values())
    merged_matches = sum(len(values) > 1 for values in ocr_to_annotation.values())
    return {
        "ocr_labels": labels,
        "matches": matches,
        "matched_labels": len(matched_annotations),
        "split_matches": split_matches,
        "merged_matches": merged_matches,
        "unmatched_labels": [
            index for index in range(len(annotation_tokens)) if index not in matched_annotations
        ],
        "unmatched_ocr_tokens": [
            index for index in range(len(ocr_tokens)) if index not in matched_ocr
        ],
        "alignment_coverage": len(matched_annotations) / max(1, len(annotation_tokens)),
    }


def normalize_alignment_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"[^\w\u0e00-\u0e7f]+", "", normalized, flags=re.UNICODE)


def _text_score(source: str, ocr: str) -> tuple[float, str]:
    if source == ocr:
        return 1.0, "exact"
    if len(source) >= 2 and source in ocr:
        return min(0.98, len(source) / max(1, len(ocr)) + 0.25), "ocr_contains_annotation"
    if len(ocr) >= 2 and ocr in source:
        return min(0.98, len(ocr) / max(1, len(source)) + 0.25), "annotation_contains_ocr"
    return SequenceMatcher(None, source, ocr).ratio(), "similarity"


def _geometry_score(left: Any, right: Any) -> float:
    try:
        lx0, ly0, lx1, ly1 = map(float, left)
        rx0, ry0, rx1, ry1 = map(float, right)
    except (TypeError, ValueError):
        return 0.0
    intersection = max(0.0, min(lx1, rx1) - max(lx0, rx0)) * max(0.0, min(ly1, ry1) - max(ly0, ry0))
    left_area = max(0.0, lx1 - lx0) * max(0.0, ly1 - ly0)
    right_area = max(0.0, rx1 - rx0) * max(0.0, ry1 - ry0)
    union = left_area + right_area - intersection
    iou = intersection / max(1e-6, union)
    lcx, lcy = (lx0 + lx1) / 2, (ly0 + ly1) / 2
    rcx, rcy = (rx0 + rx1) / 2, (ry0 + ry1) / 2
    scale = max(1.0, (left_area**0.5 + right_area**0.5) / 2)
    distance_score = max(0.0, 1.0 - ((lcx - rcx) ** 2 + (lcy - rcy) ** 2) ** 0.5 / (3 * scale))
    return max(iou, distance_score)


def _resolve_label(labels: Sequence[str]) -> str:
    counts = Counter(labels)
    priority = {"ANSWER": 7, "QUESTION": 6, "VALUE": 5, "KEY": 4, "HEADER": 3, "TABLE_CELL": 2, "OTHER": 1}
    return max(counts, key=lambda label: (counts[label], priority.get(label, 0)))
