"""Structured, evidence-bearing information extraction components."""

from .schema import CANONICAL_FIELDS, build_document_result, validate_document_result

__all__ = ["CANONICAL_FIELDS", "build_document_result", "validate_document_result"]
