"""Dataset discovery and identification.

Datasets are identified by a combination of directory names, known internal
folder structures, known annotation filenames, and file extensions. A
confidence level (high/medium/low) is recorded for every identification so
that low-confidence cases can be left in place rather than moved blindly.

Gmail documents are classified by filename keyword only (contents are never
inspected at this stage) into receipt / invoice / legal_financial / unclassified.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from . import config as cfgmod
from . import privacy

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DatasetInfo:
    name: str                       # sroie | funsd | fatura | coru | gmail
    source_type: str                # public | private
    current_path: Path              # where the data currently lives
    target_path: Path               # canonical location under data/raw/...
    confidence: str                 # high | medium | low
    evidence: list[str] = field(default_factory=list)
    internal_folders: list[str] = field(default_factory=list)

    @property
    def is_resolved(self) -> bool:
        return self.confidence in ("high", "medium") and self.current_path.exists()


@dataclass
class GmailClassification:
    category: str                   # receipt | invoice | legal_financial | unclassified
    confidence: str                 # high | medium | low
    reason: str


# ---------------------------------------------------------------------------
# Signature definitions
# ---------------------------------------------------------------------------
# Each signature lists (a) name hints and (b) structural/content markers found
# anywhere beneath a candidate root. More markers => higher confidence.

_SIGNATURES = {
    "sroie": {
        "name_hints": ("sroie", "sroie2019"),
        "markers": (
            "SROIE2019",
            "box/entities/img",          # canonical sroie split layout
            "layoutlm-base-uncased",
        ),
        "files": ("X00016469670",),      # well-known sroie receipt id family
        "extensions": (".jpg", ".txt"),
    },
    "funsd": {
        "name_hints": ("funsd",),
        "markers": (
            "training_data/annotations",
            "testing_data/annotations",
            "training_data/images",
        ),
        "files": (),
        "extensions": (".png", ".json"),
    },
    "fatura": {
        "name_hints": ("fatura",),
        "markers": (
            "Annotations/COCO_compatible_format",
            "Annotations/layoutlm_HF_format",
            "Annotations/Original_Format",
            "invoices_dataset_final",
        ),
        "files": (),
        "extensions": (".jpg",),
    },
    "coru": {
        "name_hints": ("coru",),
        "markers": (
            "Item Information Extraction",
            "OCR Dataset",
            "Receipt Images & Key Information Detection",
            "Receipt Question Answering",
        ),
        "files": (),
        "extensions": (".jpg", ".txt", ".csv", ".json"),
    },
}

# Gmail filename keyword tables (lower-cased substring match).
RECEIPT_TERMS = ("receipt", "rcpt", "ใบเสร็จ")
INVOICE_TERMS = ("invoice", "inv-", "e-invoice", "tax_invoice", "ใบกำกับภาษี")
LEGAL_FINANCIAL_TERMS = (
    "agreement", "terms", "policy", "risk", "disclosure", "regulation",
    "fee", "complaint", "privacy", "execution", "margin", "declaration",
    "regulations", "disclaimer", "conflicts",
)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_datasets(cfg: Mapping[str, Any]) -> list[DatasetInfo]:
    """Discover all known datasets under the configured candidate roots.

    Returns one DatasetInfo per detected dataset. A dataset is detected at the
    first candidate root that yields medium-or-higher confidence. The target
    path always reflects the canonical config location regardless of where the
    data currently lives.
    """
    roots = [
        cfgmod.resolve_path(cfg, root)
        for root in cfg.get("discovery", {}).get("candidate_roots", [])
    ]
    found: list[DatasetInfo] = []

    for name in ("sroie", "funsd", "fatura", "coru"):
        info = _find_public_dataset(cfg, name, roots)
        if info is not None:
            found.append(info)

    gmail_info = _find_gmail(cfg, roots)
    if gmail_info is not None:
        found.append(gmail_info)

    return found


def _find_public_dataset(
    cfg: Mapping[str, Any], name: str, roots: list[Path]
) -> DatasetInfo | None:
    sig = _SIGNATURES[name]
    target = cfgmod.resolve_path(cfg, name)

    # If the canonical target already holds the dataset (post-organization),
    # trust it with high confidence.
    if target.exists() and _has_any_marker(target, sig):
        confidence, evidence, folders = _score_candidate(target, sig)
        return DatasetInfo(
            name=name,
            source_type="public",
            current_path=target,
            target_path=target,
            confidence="high",
            evidence=["located at canonical target"] + evidence,
            internal_folders=folders,
        )

    # Otherwise search the candidate roots.
    best: tuple[str, Path, list[str], list[str]] | None = None
    for root in roots:
        if not root.exists():
            continue
        candidate = _locate_named_child(root, sig["name_hints"])
        if candidate is None:
            continue
        confidence, evidence, folders = _score_candidate(candidate, sig)
        rank = {"high": 2, "medium": 1, "low": 0}[confidence]
        if best is None or rank > {"high": 2, "medium": 1, "low": 0}[best[0]]:
            best = (confidence, candidate, evidence, folders)
        if confidence == "high":
            break

    if best is None:
        return None
    confidence, candidate, evidence, folders = best
    return DatasetInfo(
        name=name,
        source_type="public",
        current_path=candidate,
        target_path=target,
        confidence=confidence,
        evidence=evidence,
        internal_folders=folders,
    )


def _find_gmail(cfg: Mapping[str, Any], roots: list[Path]) -> DatasetInfo | None:
    """Locate the private Gmail tree, if present, under any candidate root."""
    if not cfg.get("privacy", {}).get("gmail_is_private", True):
        return None
    target = cfgmod.resolve_path(cfg, "gmail_receipts").parent  # data/raw/private/gmail

    if target.exists() and any(target.glob("**/*.pdf")):
        return DatasetInfo(
            name="gmail",
            source_type="private",
            current_path=target,
            target_path=target,
            confidence="high",
            evidence=["gmail pdfs at canonical target"],
            internal_folders=_child_dirs(target),
        )

    for root in roots:
        if not root.exists():
            continue
        # Look for a "gmail*" directory anywhere beneath the root (shallow).
        for match in list(root.glob("**/gmail*")):
            if match.is_dir() and any(match.glob("**/*.pdf")):
                return DatasetInfo(
                    name="gmail",
                    source_type="private",
                    current_path=match,
                    target_path=target,
                    confidence="high",
                    evidence=[f"found gmail tree with PDFs at {match}"],
                    internal_folders=_child_dirs(match),
                )
    return None


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _locate_named_child(root: Path, hints: tuple[str, ...]) -> Path | None:
    """Find a descendant directory whose name contains any hint (substring, case-insensitive)."""
    lowered = tuple(h.lower() for h in hints)

    def matches(name: str) -> bool:
        n = name.lower()
        return any(h in n for h in lowered)

    if root.exists():
        for child in root.iterdir():
            if child.is_dir() and matches(child.name):
                return child
        # Shallow glob (depth-limited) to keep this fast on huge trees.
        for match in root.glob("*/*"):
            if match.is_dir() and matches(match.name):
                return match
    return None


def _score_candidate(
    candidate: Path, sig: Mapping[str, Any]
) -> tuple[str, list[str], list[str]]:
    """Score a candidate directory against a signature.

    Returns (confidence, evidence, internal_folders). Confidence is high when
    structural markers are present, medium on name+extension match only, low
    otherwise.
    """
    evidence: list[str] = []
    folders = _child_dirs(candidate)
    marker_hits = 0
    for marker in sig["markers"]:
        normalized = marker.replace("/", "\\") if "\\" in str(candidate) else marker
        # Check both the marker string as a path and as a literal fragment.
        if _path_marker_present(candidate, marker) or _text_marker_present(candidate, marker, normalized):
            marker_hits += 1
            evidence.append(f"marker:{marker}")

    ext_hits = _count_extensions(candidate, sig["extensions"])
    if ext_hits:
        evidence.append(f"extensions:{ext_hits}")

    name_lowered = candidate.name.lower()
    name_hints = tuple(h.lower() for h in sig["name_hints"])
    name_match = any(h in name_lowered for h in name_hints)
    if name_match:
        evidence.append(f"name-hint:{candidate.name}")

    if marker_hits >= 1:
        confidence = "high"
    elif name_match and ext_hits:
        confidence = "medium"
    else:
        confidence = "low"
    return confidence, evidence, folders


def _path_marker_present(root: Path, marker: str) -> bool:
    """True if the marker (e.g. 'training_data/annotations') exists under root."""
    if "/" not in marker:
        return (root / marker).exists() or any(
            d.name == marker for d in _walk_dirs(root, max_depth=2)
        )
    return (root / marker.replace("/", "\\" if "\\" in str(root) else "/")).exists() or (
        root / marker
    ).exists()


def _text_marker_present(root: Path, marker: str, _normalized: str) -> bool:
    """Fallback: any directory at depth<=2 whose name matches the last segment."""
    last = marker.split("/")[-1].lower()
    return any(d.name.lower() == last for d in _walk_dirs(root, max_depth=2))


def _walk_dirs(root: Path, max_depth: int = 2) -> list[Path]:
    """Yield directories up to max_depth (bounded for performance)."""
    out: list[Path] = []
    if not root.exists():
        return out
    frontier = [root]
    for _ in range(max_depth):
        nxt: list[Path] = []
        for d in frontier:
            try:
                for child in d.iterdir():
                    if child.is_dir():
                        out.append(child)
                        nxt.append(child)
            except (PermissionError, OSError):
                continue
        frontier = nxt
    return out


def _child_dirs(root: Path) -> list[str]:
    if not root.exists():
        return []
    try:
        return sorted(p.name for p in root.iterdir() if p.is_dir())
    except (PermissionError, OSError):
        return []


def _count_extensions(root: Path, exts: tuple[str, ...]) -> int:
    """Count files matching any of the extensions (bounded shallow scan)."""
    if not root.exists():
        return 0
    lowered = tuple(e.lower() for e in exts)
    total = 0
    try:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in lowered:
                total += 1
                if total >= 50:  # enough evidence; cap for speed
                    return total
    except (PermissionError, OSError):
        return total
    return total


def _has_any_marker(root: Path, sig: Mapping[str, Any]) -> bool:
    confidence, _, _ = _score_candidate(root, sig)
    return confidence in ("high", "medium")


# ---------------------------------------------------------------------------
# Gmail classification (filename only)
# ---------------------------------------------------------------------------


def classify_gmail_filename(filename: str) -> GmailClassification:
    """Classify a Gmail document by filename keywords only.

    Order matters: legal/financial terms are checked first because filenames
    like 'Fee_detail_...pdf' would otherwise be ambiguous. Pure keyword hits
    are 'medium' confidence; a hit plus a recognized category subfolder name
    elsewhere is treated as 'high' by callers that have that context.
    """
    name = filename.lower()

    def has(terms: tuple[str, ...]) -> bool:
        return any(t in name for t in terms)

    if has(LEGAL_FINANCIAL_TERMS):
        hit = next(t for t in LEGAL_FINANCIAL_TERMS if t in name)
        return GmailClassification("legal_financial", "high", f"keyword '{hit}'")
    if has(INVOICE_TERMS):
        hit = next(t for t in INVOICE_TERMS if t in name)
        return GmailClassification("invoice", "high", f"keyword '{hit}'")
    if has(RECEIPT_TERMS):
        hit = next(t for t in RECEIPT_TERMS if t in name)
        return GmailClassification("receipt", "high", f"keyword '{hit}'")

    return GmailClassification(
        "unclassified", "low", "no category keyword matched in filename"
    )


def gmail_target_subfolder(category: str, cfg: Mapping[str, Any]) -> Path:
    """Map a Gmail category to its canonical target subfolder path."""
    mapping = {
        "receipt": cfgmod.resolve_path(cfg, "gmail_receipts"),
        "invoice": cfgmod.resolve_path(cfg, "gmail_invoices"),
        "legal_financial": cfgmod.resolve_path(cfg, "gmail_legal_financial"),
        "unclassified": cfgmod.resolve_path(cfg, "gmail_unclassified"),
    }
    return mapping[category]


# Re-export privacy helpers for convenience.
__all__ = [
    "DatasetInfo",
    "GmailClassification",
    "discover_datasets",
    "classify_gmail_filename",
    "gmail_target_subfolder",
    "privacy",
]
