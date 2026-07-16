"""Versioned inference output construction and JSON Schema validation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator

SCHEMA_VERSION = "1.0"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA_PATH = PROJECT_ROOT / "schemas" / "inference_output.schema.json"
CANONICAL_FIELDS = (
    "organization_name",
    "vendor_name",
    "customer_name",
    "person_name",
    "document_title",
    "website",
    "date",
    "time",
    "invoice_number",
    "receipt_number",
    "reference_number",
    "tax_identification_number",
    "customer_account_number",
    "subtotal",
    "discount",
    "service_charge",
    "tax",
    "total_amount",
    "paid_amount",
    "balance",
    "currency",
    "payment_method",
    "bank_name",
    "account_reference",
    "address",
    "email",
    "phone_number",
)


class OutputValidationError(ValueError):
    """Raised when an inference result violates the public contract."""


def load_output_schema(path: str | Path = DEFAULT_SCHEMA_PATH) -> dict[str, Any]:
    schema_path = Path(path)
    with schema_path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    Draft202012Validator.check_schema(schema)
    return schema


def validate_document_result(
    payload: Mapping[str, Any], schema_path: str | Path = DEFAULT_SCHEMA_PATH
) -> None:
    """Validate one complete inference result and raise a bounded error."""
    validator = Draft202012Validator(load_output_schema(schema_path))
    errors = sorted(validator.iter_errors(dict(payload)), key=lambda error: list(error.path))
    if errors:
        details = []
        for error in errors[:10]:
            location = ".".join(str(item) for item in error.absolute_path) or "$"
            details.append(f"{location}: {error.message}")
        suffix = f" (+{len(errors) - 10} more)" if len(errors) > 10 else ""
        raise OutputValidationError("; ".join(details) + suffix)


def empty_fields() -> dict[str, None]:
    """Return every canonical field explicitly set to null."""
    return {field: None for field in CANONICAL_FIELDS}


def build_document_result(
    *,
    document_id: str,
    source_type: str,
    pages: list[dict[str, Any]],
    device: str,
    duration_seconds: float = 0.0,
    private_output: bool = False,
    document_type: str = "unknown",
    document_type_confidence: float | None = None,
    selected_language_route: str = "unknown",
    detected_languages: list[str] | None = None,
    language_confidence: float | None = None,
    fields: Mapping[str, Any] | None = None,
    rotation_display: Mapping[str, Any] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Build a complete result with null-safe canonical fields."""
    canonical = empty_fields()
    if fields:
        unknown = set(fields) - set(CANONICAL_FIELDS)
        if unknown:
            raise ValueError(f"unsupported canonical fields: {sorted(unknown)}")
        canonical.update(fields)
    display = {
        "cluster_id": None,
        "zone": None,
        "confidence": None,
        "purpose": "display_only",
        "warning": "K-Means display branch did not run",
    }
    if rotation_display:
        display.update(rotation_display)
    result = {
        "schema_version": SCHEMA_VERSION,
        "document_id": str(document_id),
        "source_type": source_type,
        "document_type": {
            "label": document_type,
            "confidence": document_type_confidence,
        },
        "language": {
            "selected_route": selected_language_route,
            "detected_languages": sorted(set(detected_languages or [])),
            "confidence": language_confidence,
        },
        "rotation_display": display,
        "pages": pages,
        "fields": canonical,
        "warnings": list(warnings or []),
        "processing": {
            "duration_seconds": max(0.0, float(duration_seconds)),
            "device": device,
            "pipeline_version": SCHEMA_VERSION,
            "private_output": bool(private_output),
        },
    }
    validate_document_result(result)
    return result
