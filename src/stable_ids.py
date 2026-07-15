"""Deterministic stable identifiers for files and documents.

IDs never depend on filesystem walk order. They are derived from the dataset
name plus either the file's relative path (file_id) or the document's logical
key rank within a stable sorted ordering (document_id). Raw files are never
renamed to match these IDs during this stage.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable

# Short, collision-resistant prefix length for path-derived hex digests.
_HEX_LEN = 12


def _hex(value: str, length: int = _HEX_LEN) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def file_id(dataset: str, rel_path: str) -> str:
    """Deterministic per-file id: dataset + short hash of the POSIX rel path.

    Stable across runs because it depends only on (dataset, rel_path), not on
    traversal order. Collisions are astronomically unlikely for 12 hex chars.
    """
    norm = _normalize_rel(rel_path)
    return f"{dataset}_{_hex(f'{dataset}|{norm}')}"


def document_id(dataset: str, rank: int) -> str:
    """Sequential document id like ``sroie_000001``.

    ``rank`` MUST come from a deterministic ordering (e.g. sorted document
    keys) so the same input set yields the same ids every run. Callers obtain
    ranks via :func:`assign_document_ids`.
    """
    if rank < 0:
        raise ValueError("rank must be non-negative")
    return f"{dataset}_{rank:06d}"


def assign_document_ids(dataset: str, document_keys: Iterable[str]) -> dict[str, str]:
    """Map each deterministic document key to a stable sequential document_id.

    document_keys are sorted lexicographically before numbering, so re-running
    on the same set produces identical ids. Adding a file shifts only the ids
    after it (an accepted trade-off for human-readable sequential ids).
    """
    mapping: dict[str, str] = {}
    for rank, key in enumerate(sorted(set(document_keys))):
        mapping[key] = document_id(dataset, rank)
    return mapping


def detect_collisions(ids: Iterable[str]) -> dict[str, int]:
    """Return any id that appears more than once mapped to its count."""
    counts: dict[str, int] = {}
    for identifier in ids:
        counts[identifier] = counts.get(identifier, 0) + 1
    return {k: v for k, v in counts.items() if v > 1}


def _normalize_rel(rel_path: str) -> str:
    """Normalize a relative path to forward-slash POSIX form."""
    return rel_path.replace("\\", "/").lstrip("./")
