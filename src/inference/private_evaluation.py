"""Privacy-preserving helpers for local-only private document evaluation."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.inference.document_io import SUPPORTED_IMAGE_EXTENSIONS
from src.rotation_common import deterministic_rank


SUPPORTED_PRIVATE_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS | {".pdf"}


def discover_private_documents(
    input_root: str | Path,
    *,
    explicit_files: Sequence[str] = (),
    recursive: bool = False,
    limit: int = 0,
) -> list[Path]:
    """Select supported local files without carrying source names into reports."""
    root = Path(input_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"private input root is missing: {root}")
    if limit < 0:
        raise ValueError("private test limit must be non-negative")
    if explicit_files:
        candidates = []
        for value in explicit_files:
            path = Path(value)
            path = path.resolve() if path.is_absolute() else (root / path).resolve()
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise ValueError("--file must remain within --input-root") from exc
            candidates.append(path)
    else:
        iterator = root.rglob("*") if recursive else root.glob("*")
        candidates = list(iterator)
    usable = [
        path for path in candidates
        if path.is_file() and path.suffix.casefold() in SUPPORTED_PRIVATE_EXTENSIONS
    ]
    unique = sorted(
        set(usable),
        key=lambda path: deterministic_rank(path.relative_to(root).as_posix(), 42),
    )
    return unique[:limit] if limit else unique


def anonymous_document_id(index: int) -> str:
    if index < 1:
        raise ValueError("anonymous document index must be positive")
    return f"private_{index:06d}"


def manual_review_rows(
    anonymous_id: str, result: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Create the private, user-editable field review rows for one result."""
    document_type = str(result.get("document_type", {}).get("label", "unknown"))
    rows: list[dict[str, Any]] = []
    for name, prediction in sorted((result.get("fields") or {}).items()):
        if not isinstance(prediction, Mapping) or prediction.get("value") is None:
            continue
        rows.append({
            "anonymous_document_id": anonymous_id,
            "predicted_document_type": document_type,
            "extracted_field": name,
            "predicted_value": prediction.get("value"),
            "confidence": prediction.get("confidence"),
            "evidence_page": prediction.get("page_number"),
            "user_corrected_value": "",
            "correct_yes_no": "",
            "notes": "",
        })
    return rows


def aggregate_private_results(
    results: Sequence[Mapping[str, Any]],
    *,
    attempted_documents: int,
    error_type_counts: Mapping[str, int],
    elapsed_seconds: float,
    checkpoint_model_sha256: str | None,
) -> dict[str, Any]:
    """Return aggregate operational metrics that contain no names or OCR text."""
    counts: Counter[str] = Counter()
    confidences: list[float] = []
    durations: list[float] = []
    for result in results:
        pages = list(result.get("pages") or [])
        counts["successful_documents"] += 1
        counts["pages"] += len(pages)
        counts["non_null_fields"] += sum(
            isinstance(value, Mapping) and value.get("value") is not None
            for value in (result.get("fields") or {}).values()
        )
        counts[f"document_type:{result.get('document_type', {}).get('label', 'unknown')}"] += 1
        for page in pages:
            ocr = page.get("ocr") or {}
            counts["ocr_words"] += len(ocr.get("words") or [])
            counts["entities"] += len(page.get("entities") or [])
            counts["relations"] += len(page.get("key_value_pairs") or [])
            counts[f"route:{ocr.get('language_route', 'unknown')}"] += 1
            if ocr.get("mean_confidence") is not None:
                confidences.append(float(ocr["mean_confidence"]))
        duration = result.get("processing", {}).get("duration_seconds")
        if duration is not None:
            durations.append(float(duration))
    successful = counts["successful_documents"]
    return {
        "schema_version": "1.0",
        "status": "private_test_aggregate",
        "attempted_documents": int(attempted_documents),
        "successful_documents": successful,
        "failed_documents": int(attempted_documents) - successful,
        "processed_pages": counts["pages"],
        "mean_ocr_words_per_page": counts["ocr_words"] / max(1, counts["pages"]),
        "mean_entities_per_page": counts["entities"] / max(1, counts["pages"]),
        "mean_relations_per_page": counts["relations"] / max(1, counts["pages"]),
        "mean_non_null_fields_per_document": counts["non_null_fields"] / max(1, successful),
        "mean_ocr_confidence": sum(confidences) / len(confidences) if confidences else None,
        "mean_duration_seconds": sum(durations) / len(durations) if durations else None,
        "route_counts": {
            key.split(":", 1)[1]: value
            for key, value in sorted(counts.items()) if key.startswith("route:")
        },
        "document_type_counts": {
            key.split(":", 1)[1]: value
            for key, value in sorted(counts.items()) if key.startswith("document_type:")
        },
        "error_type_counts": dict(sorted(error_type_counts.items())),
        "checkpoint_model_sha256": checkpoint_model_sha256,
        "gmail_fit_rows": 0,
        "local_processing_only": True,
        "contains_filenames": False,
        "contains_ocr_text": False,
        "contains_images": False,
        "contains_per_document_predictions": False,
        "elapsed_seconds": float(elapsed_seconds),
        "limitations": [
            "No private ground truth is used; this is aggregate operational testing, not an accuracy estimate."
        ],
    }


def aggregate_markdown(report: Mapping[str, Any]) -> str:
    return "\n".join([
        "# Private Gmail aggregate operational test",
        "",
        f"Attempted documents: {report['attempted_documents']}.",
        f"Successful documents: {report['successful_documents']}.",
        f"Failed documents: {report['failed_documents']}.",
        f"Processed pages: {report['processed_pages']}.",
        f"Mean non-null fields per successful document: {report['mean_non_null_fields_per_document']:.3f}.",
        "Gmail fit rows: **0**.",
        "",
        "This report contains aggregate counts only. It contains no filenames, OCR text, images, or per-document predictions.",
        "It is an operational check because no private ground truth was used.",
    ]) + "\n"
