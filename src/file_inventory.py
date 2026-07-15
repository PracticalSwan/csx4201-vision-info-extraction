"""File inventory construction and per-file validation.

Walks each discovered dataset, validates images / PDFs / annotations WITHOUT
modifying them, computes SHA-256, and emits one inventory row per file plus a
processing-error row for anything unreadable or malformed. Raw files are never
opened for writing and never rewritten.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import mimetypes
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import dataset_discovery as dd
from . import privacy
from . import stable_ids

log = logging.getLogger("vix.inventory")

INVENTORY_COLUMNS = [
    "file_id",
    "document_id",
    "dataset",
    "source_type",
    "document_category",
    "original_filename",
    "current_relative_path",
    "original_relative_path",
    "extension",
    "mime_type",
    "size_bytes",
    "modified_time",
    "sha256",
    "is_private",
    "is_annotation",
    "is_image",
    "is_pdf",
    "is_readable",
    "is_empty",
    "classification_confidence",
    "notes",
]

PROCESSING_ERROR_COLUMNS = [
    "timestamp",
    "dataset",
    "file_path",
    "operation",
    "error_type",
    "error_message",
    "action_taken",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".webp"}
PDF_EXTS = {".pdf"}
JSON_EXTS = {".json"}
CSV_EXTS = {".csv"}
TXT_EXTS = {".txt"}
XML_EXTS = {".xml"}
ARCHIVE_EXTS = {".zip", ".rar", ".7z", ".tar", ".gz", ".tgz"}
MODEL_EXTS = {".bin"}  # e.g. bundled layoutlm weights — flagged, out of scope
ANNOTATION_PATH_HINTS = ("annotations", "box", "entities", "labels", "annotation")


@dataclass
class InventoryResult:
    rows: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)

    def add_error(self, dataset: str, path: Path, operation: str,
                  error_type: str, message: str, action: str = "recorded") -> None:
        self.errors.append({
            "timestamp": _now_iso(),
            "dataset": dataset,
            "file_path": str(path),
            "operation": operation,
            "error_type": error_type,
            "error_message": message,
            "action_taken": action,
        })


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_inventory(
    datasets: list["dd.DatasetInfo"],
    cfg: Mapping[str, Any],
    progress: bool = False,
) -> InventoryResult:
    """Build the full file inventory across all discovered datasets."""
    result = InventoryResult()
    chunk = int(cfg.get("audit", {}).get("sha256_chunk_size", 1 << 20))
    do_sha = bool(cfg.get("audit", {}).get("calculate_sha256", True))
    inspect_images = bool(cfg.get("audit", {}).get("inspect_images", True))
    inspect_pdfs = bool(cfg.get("audit", {}).get("inspect_pdfs", True))
    inspect_ann = bool(cfg.get("audit", {}).get("inspect_annotations", True))

    for ds in datasets:
        if not ds.current_path.exists():
            result.add_error(
                ds.name, ds.current_path, "discover", "permission_error",
                f"dataset root missing: {ds.current_path}", "skipped dataset",
            )
            continue
        _inventory_dataset(
            ds, cfg, result, chunk=chunk, do_sha=do_sha,
            inspect_images=inspect_images, inspect_pdfs=inspect_pdfs,
            inspect_ann=inspect_ann, progress=progress,
        )

    _assign_document_ids(result.rows)
    return result


# ---------------------------------------------------------------------------
# Per-dataset walk
# ---------------------------------------------------------------------------


def _inventory_dataset(
    ds: "dd.DatasetInfo",
    cfg: Mapping[str, Any],
    result: InventoryResult,
    *,
    chunk: int,
    do_sha: bool,
    inspect_images: bool,
    inspect_pdfs: bool,
    inspect_ann: bool,
    progress: bool,
) -> None:
    root = ds.current_path
    files = list(_iter_files(root))
    workers = max(1, int(cfg.get("runtime", {}).get("workers", 2)))
    build_kwargs = dict(chunk=chunk, do_sha=do_sha, inspect_images=inspect_images,
                        inspect_pdfs=inspect_pdfs, inspect_ann=inspect_ann)

    # File hashing and image decoding release the GIL, so a thread pool gives a
    # real speedup on large corpora. Each task is independent; results are
    # sorted afterwards for a stable, deterministic inventory order.
    from concurrent.futures import ThreadPoolExecutor, as_completed

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_path = {
            pool.submit(_build_row, p, root, ds, cfg, **build_kwargs): p for p in files
        }
        try:
            from tqdm import tqdm
            iterator = tqdm(as_completed(future_to_path), total=len(future_to_path),
                            desc=ds.name, unit="file")
        except Exception:  # pragma: no cover
            iterator = as_completed(future_to_path)

        for fut in iterator:
            path = future_to_path[fut]
            try:
                rows.append(fut.result())
            except Exception as exc:  # never let one file abort the audit
                if not cfg.get("audit", {}).get("continue_on_error", True):
                    raise
                result.add_error(ds.name, path, "inventory", "unreadable_file",
                                 f"{type(exc).__name__}: {exc}", "recorded; skipped")
                rows.append(_skeleton_row(path, root, ds, cfg, note=f"inventory error: {exc}"))

    rows.sort(key=lambda r: r["current_relative_path"])
    result.rows.extend(rows)


def _iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file():
            yield p


def _build_row(
    path: Path, root: Path, ds: "dd.DatasetInfo", cfg: Mapping[str, Any],
    *, chunk: int, do_sha: bool, inspect_images: bool, inspect_pdfs: bool,
    inspect_ann: bool,
) -> dict[str, Any]:
    rel = path.relative_to(root).as_posix()
    ext = path.suffix.lower()
    stat = path.stat()
    size = stat.st_size
    is_empty = size == 0
    notes: list[str] = []

    is_image = ext in IMAGE_EXTS
    is_pdf = ext in PDF_EXTS
    is_annotation = _looks_like_annotation(path, ext)
    is_archive = ext in ARCHIVE_EXTS
    is_model = ext in MODEL_EXTS

    is_readable = True
    sha = ""
    width = height = None
    page_count = None

    if is_empty:
        is_readable = False
        notes.append("empty file")

    if do_sha and not is_empty:
        sha = _sha256(path, chunk) or ""
        if sha is None:
            is_readable = False
            notes.append("sha256 read failed")

    # Content-type validation (read-only; never rewrite).
    if is_image and inspect_images and not is_empty:
        wh = _inspect_image(path)
        if wh is None:
            is_readable = False
            notes.append("corrupted or unreadable image")
        else:
            width, height = wh
            if width <= 0 or height <= 0:
                is_readable = False
                notes.append("non-positive image dimensions")
    if is_pdf and inspect_pdfs and not is_empty:
        pc = _inspect_pdf(path)
        if pc is None:
            is_readable = False
            notes.append("corrupted or unreadable pdf")
        else:
            page_count = pc
    if inspect_ann and not is_empty:
        ann_note = _inspect_annotation(path, ext)
        if ann_note is not None:
            if ann_note.startswith("invalid"):
                is_readable = False
            notes.append(ann_note)

    if is_archive:
        notes.append("archive (not extracted)")
    if is_model:
        notes.append("pretrained model artifact (out of scope; preserved as-is)")
    if path.name.lower() == ".ds_store" or "__macosx" in rel.lower():
        notes.append("OS junk")

    doc_cat = _document_category(ds.name, path, ext)
    confidence = ds.confidence if ds.name != "gmail" else _gmail_confidence_for_row(path)

    row = {
        "file_id": stable_ids.file_id(ds.name, rel),
        "document_id": "",  # filled by _assign_document_ids
        "dataset": ds.name,
        "source_type": ds.source_type,
        "document_category": doc_cat,
        "original_filename": path.name,
        "current_relative_path": f"{ds.name}/{rel}",
        "original_relative_path": rel,
        "extension": ext,
        "mime_type": _guess_mime(path, ext),
        "size_bytes": size,
        "modified_time": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "sha256": sha,
        "is_private": privacy.is_private(path, cfg),
        "is_annotation": is_annotation,
        "is_image": is_image,
        "is_pdf": is_pdf,
        "is_readable": is_readable,
        "is_empty": is_empty,
        "classification_confidence": confidence,
        "notes": "; ".join(notes),
    }
    # Stash extra validated metadata internally (not in CSV columns) for
    # downstream summary/duplicate steps.
    row["_width"] = width
    row["_height"] = height
    row["_page_count"] = page_count
    row["_doc_key"] = _document_key(ds.name, path)
    row["_abs_path"] = str(path)
    return row


def _skeleton_row(path: Path, root: Path, ds, cfg, note: str) -> dict[str, Any]:
    rel = path.relative_to(root).as_posix() if root in path.parents or path == root else path.name
    try:
        rel = path.relative_to(root).as_posix()
    except ValueError:
        rel = path.name
    stat = path.stat() if path.exists() else None
    return {
        "file_id": stable_ids.file_id(ds.name, rel),
        "document_id": "",
        "dataset": ds.name,
        "source_type": ds.source_type,
        "document_category": "unknown",
        "original_filename": path.name,
        "current_relative_path": f"{ds.name}/{rel}",
        "original_relative_path": rel,
        "extension": path.suffix.lower(),
        "mime_type": _guess_mime(path, path.suffix.lower()),
        "size_bytes": stat.st_size if stat else 0,
        "modified_time": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat() if stat else "",
        "sha256": "",
        "is_private": privacy.is_private(path, cfg),
        "is_annotation": False,
        "is_image": False,
        "is_pdf": False,
        "is_readable": False,
        "is_empty": (stat.st_size if stat else 0) == 0,
        "classification_confidence": "low",
        "notes": note,
        "_doc_key": _document_key(ds.name, path),
        "_abs_path": str(path),
        "_width": None,
        "_height": None,
        "_page_count": None,
    }


# ---------------------------------------------------------------------------
# Document id assignment (stable, sorted)
# ---------------------------------------------------------------------------


def _assign_document_ids(rows: list[dict[str, Any]]) -> None:
    by_dataset: dict[str, set[str]] = {}
    for r in rows:
        by_dataset.setdefault(r["dataset"], set()).add(r.get("_doc_key") or r["original_filename"])
    mappings: dict[str, dict[str, str]] = {}
    for ds_name, keys in by_dataset.items():
        mappings[ds_name] = stable_ids.assign_document_ids(ds_name, keys)
    for r in rows:
        key = r.get("_doc_key") or r["original_filename"]
        r["document_id"] = mappings[r["dataset"]].get(key, "")


def _document_key(dataset: str, path: Path) -> str:
    """Logical document key: usually the filename stem, so paired image +
    annotation files collapse to one document."""
    if dataset == "coru":
        # CORU has parallel components; scope the key by top-level component so
        # identically-named crops across components don't collide.
        parts = path.parts
        comp = parts[0] if parts else ""
        return f"{comp}/{path.stem}"
    return path.stem


# ---------------------------------------------------------------------------
# Category + annotation heuristics
# ---------------------------------------------------------------------------


def _document_category(dataset: str, path: Path, ext: str) -> str:
    if dataset == "gmail":
        return "unknown"  # refined by classifier at audit layer using filename
    if dataset == "sroie":
        return "receipt"
    if dataset == "funsd":
        return "form"
    if dataset == "fatura":
        return "invoice"
    if dataset == "coru":
        rel = path.as_posix().lower()
        if "ocr dataset" in rel:
            return "OCR_annotation"
        if "item information extraction" in rel:
            return "field_annotation"
        if "receipt question answering" in rel:
            return "question_answer"
        if "receipt images & key information detection" in rel:
            return "field_annotation" if ext in (".txt", ".json") else "receipt"
        return "receipt"
    return "unknown"


def _looks_like_annotation(path: Path, ext: str) -> bool:
    if ext in JSON_EXTS or ext in XML_EXTS or ext in CSV_EXTS:
        return True
    if ext in TXT_EXTS:
        return any(hint in path.as_posix().lower() for hint in ANNOTATION_PATH_HINTS)
    return False


def _gmail_confidence_for_row(path: Path) -> str:
    # Reflect filename-classification confidence for private files.
    parent = path.parent.name.lower()
    if parent in ("receipts", "invoices", "legal_financial_docs"):
        return "high"
    return "medium"


# ---------------------------------------------------------------------------
# Readers (read-only)
# ---------------------------------------------------------------------------


def _sha256(path: Path, chunk: int) -> str | None:
    h = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(chunk), b""):
                h.update(block)
    except (OSError, PermissionError):
        return None
    return h.hexdigest()


def _inspect_image(path: Path) -> tuple[int, int] | None:
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        return None
    try:
        # Header + integrity check only: Image.open reads the header (so .size
        # is available) and .verify() checks that the decoder can parse the file
        # without fully decoding pixel data. This is fast on large images and
        # still detects truncated/corrupt headers and bad markers. We never
        # write the file back.
        with Image.open(path) as im:
            width, height = int(im.width), int(im.height)
            im.verify()
        return width, height
    except Exception:
        return None


def _inspect_pdf(path: Path) -> int | None:
    try:
        import fitz  # PyMuPDF
    except ImportError:  # pragma: no cover
        return None
    try:
        doc = fitz.open(path)
        try:
            pages = doc.page_count
        finally:
            doc.close()
        return int(pages)
    except Exception:
        return None


def _inspect_annotation(path: Path, ext: str) -> str | None:
    if ext in JSON_EXTS:
        try:
            with path.open("r", encoding="utf-8") as fh:
                json.load(fh)
            return None
        except UnicodeDecodeError:
            try:
                with path.open("r", encoding="utf-8-sig") as fh:
                    json.load(fh)
                return None
            except Exception as exc:
                return f"invalid_json: {exc}"
        except Exception as exc:
            return f"invalid_json: {exc}"
    if ext in CSV_EXTS:
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as fh:
                reader = csv.reader(fh)
                next(reader, None)  # header probe
            return None
        except UnicodeDecodeError:
            try:
                with path.open("r", encoding="latin-1", newline="") as fh:
                    next(csv.reader(fh), None)
                return "csv:non-utf8(latin-1 fallback ok)"
            except Exception as exc:
                return f"invalid_csv: {exc}"
        except Exception as exc:
            return f"invalid_csv: {exc}"
    if ext in TXT_EXTS or ext in XML_EXTS:
        return _readability_note(path)
    return None


def _readability_note(path: Path) -> str | None:
    try:
        with path.open("rb") as fh:
            raw = fh.read(4096)
        try:
            raw.decode("utf-8")
            return None
        except UnicodeDecodeError:
            raw.decode("latin-1")
            return "txt:non-utf8(latin-1 fallback ok)"
    except Exception as exc:
        return f"unreadable: {exc}"


def _guess_mime(path: Path, ext: str) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    if mime:
        return mime
    fallback = {
        ".json": "application/json", ".txt": "text/plain", ".csv": "text/csv",
        ".xml": "application/xml", ".bin": "application/octet-stream",
        ".zip": "application/zip",
    }
    return fallback.get(ext, "application/octet-stream")


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------


def write_inventory_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=INVENTORY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_errors_csv(errors: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=PROCESSING_ERROR_COLUMNS)
        writer.writeheader()
        for r in errors:
            writer.writerow(r)
