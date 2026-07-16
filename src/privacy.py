"""Privacy rules for separating public and private (Gmail) data.

Gmail documents are private test data. Their real filenames may themselves be
sensitive, so public reports must use anonymized identifiers instead. This
module centralizes every public/private decision so it cannot drift across
scripts.
"""
from __future__ import annotations

from pathlib import Path
from collections.abc import Iterable
from typing import Any, Mapping

# Path fragments that mark a file or tree as private.
PRIVATE_MARKERS = ("gmail", "private", "legal_financial_docs")

# Substrings that, if present in a path, indicate the public/private boundary
# was crossed (a private file landed under a public tree).
PUBLIC_MARKERS = ("data/raw/public", "public_train")


def require_private_input_mode(
    inputs: Iterable[str | Path],
    private_roots: Iterable[str | Path],
    *,
    private_output: bool,
) -> bool:
    """Fail closed when a configured private input is not in private mode.

    Paths are resolved without opening the document, so symlinks and Windows
    junctions cannot bypass the configured Gmail-root boundary. The error
    intentionally omits the private filename.
    """
    roots = [Path(root).resolve(strict=False) for root in private_roots]
    has_private_input = any(
        _is_within(Path(source).resolve(strict=False), root)
        for source in inputs
        for root in roots
    )
    if has_private_input and not private_output:
        raise ValueError(
            "configured private Gmail input requires --private-output and an "
            "output under the ignored private root"
        )
    return has_private_input


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def is_private(path: str | Path, cfg: Mapping[str, Any] | None = None) -> bool:
    """True if the path belongs to private (Gmail) data.

    A path is private if it contains a private marker fragment. The Gmail flag
    in config is honoured when provided.
    """
    if cfg is not None and not cfg.get("privacy", {}).get("gmail_is_private", True):
        return False
    p = str(path).replace("\\", "/").lower()
    return any(marker in p for marker in PRIVATE_MARKERS)


def is_public(path: str | Path) -> bool:
    """True if the path lives under a public data tree."""
    p = str(path).replace("\\", "/").lower()
    return any(marker in p for marker in PUBLIC_MARKERS)


def assert_private_not_under_public(path: str | Path) -> None:
    """Raise ValueError if a private path is also under a public tree.

    Used by verification to enforce strict public/private separation.
    """
    if is_private(path) and is_public(path):
        raise ValueError(
            f"Private file leaked into public tree: {path}"
        )


def anonymize_filename(dataset: str, file_id: str, extension: str) -> str:
    """Produce a non-identifying replacement filename for private files.

    The original extension is preserved so downstream tooling still knows the
    file type, but no personal information from the real name is exposed.
    """
    ext = extension.lstrip(".").lower()
    return f"{file_id}{('.' + ext) if ext else ''}"


def safe_path_for_report(
    rel_path: str,
    dataset: str,
    file_id: str,
    cfg: Mapping[str, Any],
) -> str:
    """Return a path safe to write into a public (commitable) report.

    Public datasets keep their real relative path. Private (Gmail) paths are
    replaced by an anonymized form when the config forbids exposing private
    filenames in public reports.
    """
    if not is_private(rel_path, cfg):
        return rel_path
    if cfg.get("privacy", {}).get("include_private_filenames_in_public_reports", False):
        return rel_path
    ext = Path(rel_path).suffix
    # Keep only the category subfolder (receipts/invoices/...) plus anon name,
    # never the real filename.
    parent = Path(rel_path).parent.as_posix()
    anon = anonymize_filename(dataset, file_id, ext)
    return f"{parent}/{anon}" if parent and parent != "." else anon


def public_report_sanitized(text: str, private_filenames: list[str]) -> str:
    """Strip any known private filename string occurrences from report text."""
    sanitized = text
    for name in private_filenames:
        sanitized = sanitized.replace(name, "<private-filename>")
    return sanitized
