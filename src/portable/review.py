"""Privacy-bounded result catalog for optional Codex/GPT-5.6 review."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .results import load_result, ocr_text


MAX_CATALOG_RESULTS = 250
MAX_REVIEW_TEXT_CHARS = 4_000


class ReviewPayloadError(ValueError):
    """Raised when a result cannot be shared through the bounded review tool."""


def _is_private(result_path: Path, output_root: Path, payload: Mapping[str, Any]) -> bool:
    processing = dict(payload.get("processing") or {})
    if bool(processing.get("private_output")):
        return True
    try:
        relative_parts = result_path.relative_to(output_root).parts
    except ValueError:
        return True
    return any(
        part.casefold() in {"private", "private_outputs", "private-evaluation"}
        for part in relative_parts
    )


def _opaque_id(result_path: Path, output_root: Path, payload: Mapping[str, Any]) -> str:
    relative = result_path.relative_to(output_root).as_posix()
    material = f"{relative}\0{payload.get('document_id', '')}".encode("utf-8")
    return "doc_" + hashlib.sha256(material).hexdigest()[:20]


def _catalog_entries(output_root: str | Path) -> list[tuple[dict[str, Any], Path, dict[str, Any]]]:
    root = Path(output_root).expanduser().resolve()
    if not root.is_dir():
        return []
    candidates = sorted(
        root.rglob("document_result.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )[:MAX_CATALOG_RESULTS]
    entries = []
    for path in candidates:
        try:
            payload = load_result(path)
        except (OSError, ValueError):
            continue
        if _is_private(path, root, payload):
            continue
        document_type = payload.get("document_type") or {}
        fields = {
            name: evidence
            for name, evidence in dict(payload.get("fields") or {}).items()
            if isinstance(evidence, Mapping)
            and evidence.get("value") not in {None, ""}
        }
        metadata = {
            "document_id": _opaque_id(path, root, payload),
            "created_at": datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc
            ).isoformat(),
            "page_count": len(payload.get("pages") or []),
            "field_count": len(fields),
            "field_names": sorted(fields),
            "document_type": (
                document_type.get("label")
                if isinstance(document_type, Mapping)
                else str(document_type)
            ),
            "language_route": (
                (payload.get("language") or {}).get("selected_route")
                if isinstance(payload.get("language"), Mapping)
                else None
            ),
        }
        entries.append((metadata, path, payload))
    return entries


def list_reviewable_results(output_root: str | Path) -> dict[str, Any]:
    """List only opaque public-result metadata; never return paths or text."""
    return {
        "status": "ok",
        "results": [
            metadata
            for metadata, _path, _payload in _catalog_entries(output_root)
        ],
        "privacy": (
            "Private-marked results and private output folders are excluded. "
            "No filenames, paths, field values, or OCR text are returned."
        ),
    }


def prepare_review_payload(
    output_root: str | Path,
    document_id: str,
    *,
    confirmed_cloud_review: bool,
    selected_fields: Iterable[str] | None,
    include_ocr_text: bool = False,
    max_text_chars: int = 0,
) -> dict[str, Any]:
    """Return the smallest explicitly approved payload for GPT-5.6 review."""
    entries = _catalog_entries(output_root)
    match = next(
        (
            (metadata, payload)
            for metadata, _path, payload in entries
            if metadata["document_id"] == document_id
        ),
        None,
    )
    if match is None:
        raise ReviewPayloadError(
            "unknown or non-reviewable document ID; list results again"
        )
    metadata, payload = match
    if not confirmed_cloud_review:
        return {
            "status": "confirmation_required",
            "document_id": document_id,
            "available_field_names": metadata["field_names"],
            "message": (
                "Ask the user to confirm sending the selected field values "
                "to their Codex/GPT-5.6 session. OCR text needs separate consent."
            ),
        }
    if selected_fields is None:
        raise ReviewPayloadError(
            "selected_fields must be an explicit list after user confirmation"
        )
    selected = list(dict.fromkeys(str(name) for name in selected_fields))
    available = {
        name: evidence
        for name, evidence in dict(payload.get("fields") or {}).items()
        if isinstance(evidence, Mapping)
        and evidence.get("value") not in {None, ""}
    }
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ReviewPayloadError("unknown selected fields: " + ", ".join(unknown))
    if include_ocr_text and max_text_chars <= 0:
        raise ReviewPayloadError(
            "max_text_chars must be positive when OCR text is approved"
        )
    bounded_text_chars = min(max(0, int(max_text_chars)), MAX_REVIEW_TEXT_CHARS)
    fields = {}
    for name in selected:
        evidence = dict(available[name] or {})
        fields[name] = {
            "value": evidence.get("value"),
            "confidence": evidence.get("confidence"),
            "method": evidence.get("method") or evidence.get("extraction_source"),
            "validation_status": evidence.get("validation_status"),
            "page_number": evidence.get("page_number"),
        }
    review_payload: dict[str, Any] = {
        "status": "ready",
        "document_id": document_id,
        "document_type": metadata["document_type"],
        "page_count": metadata["page_count"],
        "fields": fields,
        "review_contract": {
            "local_result_is_authoritative": True,
            "output_is_suggestions_only": True,
            "raw_document_included": False,
            "paths_or_filenames_included": False,
            "api_key_used": False,
        },
    }
    if include_ocr_text:
        text = ocr_text(payload)
        review_payload["ocr_text"] = text[:bounded_text_chars]
        review_payload["ocr_text_truncated"] = len(text) > bounded_text_chars
    return review_payload
