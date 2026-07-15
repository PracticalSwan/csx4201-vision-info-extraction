#!/usr/bin/env python3
"""audit_data.py — full data audit: inventory, validation, duplicates, summaries.

Produces, under data/metadata/:
    file_inventory.csv          one row per file (Gmail filenames anonymized)
    private_file_inventory.csv  Gmail rows with real filenames (gitignored)
    data_sources.csv            one row per logical document (public-safe)
    dataset_summary.json        per-dataset aggregate statistics
    processing_errors.csv       every unreadable/invalid file
    duplicate_report.csv        exact + near-duplicate groups
    unmatched_files.csv         images/annotations missing their pair

Usage:
    python scripts/audit_data.py [--config config.yaml] [--limit N]
           [--skip-near-duplicates] [--log-level INFO]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Force UTF-8 output so non-ASCII filenames render on the Windows console.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config as cfgmod  # noqa: E402
from src import dataset_discovery as dd  # noqa: E402
from src import dataset_validation as dv  # noqa: E402
from src import duplicate_detection as dup  # noqa: E402
from src import file_inventory as inv  # noqa: E402
from src import privacy  # noqa: E402

log = logging.getLogger("vix.audit")

DATA_SOURCE_COLUMNS = [
    "document_id", "dataset", "source_type", "document_type", "language",
    "original_filename", "current_relative_path", "annotation_path",
    "file_format", "page_count", "size_bytes", "is_private", "is_usable",
    "has_annotation", "annotation_format", "split_status", "privacy_status",
    "notes",
]

ANN_EXTS = {".json", ".xml", ".txt", ".csv"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    ap.add_argument("--limit", type=int, default=0,
                    help="cap files inventoried per dataset (0 = no cap; debug aid)")
    ap.add_argument("--skip-near-duplicates", action="store_true")
    ap.add_argument("--log-level", default=None)
    args = ap.parse_args()

    cfg = cfgmod.load_config(args.config)
    cfgmod.setup_logging(cfg, args.log_level)
    metadata_dir = cfgmod.resolve_path(cfg, "metadata")
    metadata_dir.mkdir(parents=True, exist_ok=True)

    datasets = dd.discover_datasets(cfg)
    if not datasets:
        log.error("No datasets discovered.")
        return 1

    if args.limit:
        datasets = [_LimitedDataset.wrap(d, args.limit) for d in datasets]

    log.info("Building inventory for %d datasets...", len(datasets))
    result = inv.build_inventory(datasets, cfg, progress=True)
    rows = result.rows
    # Unreadable/empty/invalid files are flagged gracefully on the row (notes +
    # is_readable/is_empty) rather than raised. Surface them in the processing
    # error log too, so processing_errors.csv is a complete record.
    _augment_errors_from_inventory(result, rows)
    log.info("Inventory rows: %d  errors: %d", len(rows), len(result.errors))

    # Exact duplicates and pair validation are fast; compute them first.
    exact = dup.find_exact_duplicates(rows) if cfg["duplicates"]["exact_enabled"] else []
    unmatched = dv.find_unmatched(rows)
    log.info("Exact duplicate groups: %d  unmatched files: %d", len(exact), len(unmatched))

    # Write the core metadata BEFORE the (potentially slower) near-duplicate
    # scan, so the inventory is never lost if that step is interrupted.
    inv.write_inventory_csv(_public_rows(rows, cfg), metadata_dir / "file_inventory.csv")
    inv.write_inventory_csv(_private_rows(rows), metadata_dir / "private_file_inventory.csv")
    write_data_sources(rows, cfg, metadata_dir / "data_sources.csv")
    inv.write_errors_csv(result.errors, metadata_dir / "processing_errors.csv")
    dv.write_unmatched_csv(unmatched, metadata_dir / "unmatched_files.csv")
    log.info("Core metadata written.")

    # Near-duplicate scan (capped and bucket-bounded so it always finishes).
    near: list[list[dict[str, Any]]] = []
    if cfg["duplicates"]["perceptual_enabled"] and not args.skip_near_duplicates:
        log.info("Near-duplicate scan (cap=%d)...",
                 cfg["duplicates"]["max_images_for_full_near_duplicate_scan"])
        near = dup.find_near_duplicates(rows, cfg)
    dup_rows = dup.build_duplicate_report(exact, near)
    log.info("Near-duplicate groups: %d", len(near))

    # Summary + duplicate report last (they depend on near-dup results).
    write_dataset_summary(datasets, rows, exact, near, unmatched,
                          result.errors, cfg, metadata_dir / "dataset_summary.json")
    dup.write_duplicate_csv(dup_rows, metadata_dir / "duplicate_report.csv")

    log.info("Audit complete. Metadata written to %s", metadata_dir)
    return 0


def _augment_errors_from_inventory(result, rows: list[dict[str, Any]]) -> None:
    """Add processing-error rows for inventory files flagged unreadable/empty.

    The inventory records per-file validity on the row itself; mirror anything
    not readable or empty into the processing error log so it is a complete
    record. Dedup by file_path against exception-based errors already present.
    """
    existing = {e["file_path"] for e in result.errors}
    for r in rows:
        path = r.get("_abs_path") or r.get("current_relative_path", "")
        if path in existing:
            continue
        notes = r.get("notes", "") or ""
        if r.get("is_empty"):
            result.add_error(r["dataset"], Path(path), "validate", "empty_file",
                             notes or "empty file", "recorded")
            existing.add(path)
        elif r.get("is_readable") is False:
            etype = _error_type_from_notes(notes, r)
            result.add_error(r["dataset"], Path(path), "validate", etype,
                             notes or "unreadable file", "recorded")
            existing.add(path)


def _error_type_from_notes(notes: str, row: dict[str, Any]) -> str:
    n = notes.lower()
    if "invalid_json" in n:
        return "invalid_json"
    if "invalid_csv" in n:
        return "invalid_csv"
    if "corrupted or unreadable pdf" in n or "pdf" in n:
        return "corrupted_pdf"
    if "corrupted" in n and row.get("is_image"):
        return "corrupted_image"
    if "non-positive image" in n:
        return "corrupted_image"
    if "unreadable" in n:
        return "unreadable_file"
    return "unreadable_file"


# ---------------------------------------------------------------------------
# Privacy-aware row selection
# ---------------------------------------------------------------------------


def _public_rows(rows: list[dict[str, Any]], cfg) -> list[dict[str, Any]]:
    """All rows; Gmail filenames anonymized when configured."""
    out = []
    for r in rows:
        if privacy.is_private(r["current_relative_path"], cfg):
            if not cfg["privacy"].get("include_private_filenames_in_public_reports", False):
                r = r.copy()
                anon = privacy.safe_path_for_report(
                    r["current_relative_path"], r["dataset"], r["file_id"], cfg)
                r["current_relative_path"] = anon
                # Both path columns are part of the committable inventory.  The
                # original-relative field previously retained the real Gmail
                # filename even though the current-relative field was redacted.
                r["original_relative_path"] = anon
                r["original_filename"] = Path(anon).name
        out.append(r)
    return out


def _private_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("is_private")]


# ---------------------------------------------------------------------------
# data_sources.csv
# ---------------------------------------------------------------------------


def write_data_sources(rows: list[dict[str, Any]], cfg, path: Path) -> None:
    by_doc: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_doc[(r["dataset"], r["document_id"])].append(r)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=DATA_SOURCE_COLUMNS)
        writer.writeheader()
        for (dataset, doc_id), group in sorted(by_doc.items()):
            writer.writerow(_doc_to_source(dataset, doc_id, group, cfg))


def _doc_to_source(dataset: str, doc_id: str, group: list[dict[str, Any]], cfg) -> dict[str, Any]:
    primary = _primary_file(group)
    annotations = [r for r in group if r.get("is_annotation")]
    ann = annotations[0] if annotations else None
    is_private = bool(group[0].get("is_private"))
    privacy_status = "private" if is_private else "public"
    filename = primary["original_filename"]
    rel = primary["current_relative_path"]
    if is_private and not cfg["privacy"].get("include_private_filenames_in_public_reports", False):
        filename = privacy.anonymize_filename(dataset, primary["file_id"], primary["extension"])
        rel = privacy.safe_path_for_report(rel, dataset, primary["file_id"], cfg)
    ann_rel = ""
    if ann:
        ann_rel = ann["current_relative_path"]
        if is_private and not cfg["privacy"].get("include_private_filenames_in_public_reports", False):
            ann_rel = privacy.safe_path_for_report(ann_rel, dataset, ann["file_id"], cfg)
    page_count = ""
    for r in group:
        if r.get("_page_count") is not None:
            page_count = r["_page_count"]
            break
    return {
        "document_id": doc_id,
        "dataset": dataset,
        "source_type": group[0]["source_type"],
        "document_type": _doc_type(dataset, primary),
        "language": _language_hint(dataset),
        "original_filename": filename,
        "current_relative_path": rel,
        "annotation_path": ann_rel,
        "file_format": primary["extension"],
        "page_count": page_count,
        "size_bytes": sum(int(r["size_bytes"]) for r in group),
        "is_private": is_private,
        "is_usable": all(bool(r["is_readable"]) for r in group if r is primary) or primary["is_readable"],
        "has_annotation": bool(annotations),
        "annotation_format": ann["extension"] if ann else "",
        "split_status": "unassigned",
        "privacy_status": privacy_status,
        "notes": "",
    }


def _primary_file(group: list[dict[str, Any]]) -> dict[str, Any]:
    for pref in (lambda r: r.get("is_image"), lambda r: r.get("is_pdf")):
        for r in group:
            if pref(r):
                return r
    return group[0]


def _doc_type(dataset: str, primary: dict[str, Any]) -> str:
    cat = primary.get("document_category", "unknown")
    if cat != "unknown":
        return cat
    return dataset


def _language_hint(dataset: str) -> str:
    return {"sroie": "en", "funsd": "en", "fatura": "tr", "coru": "en", "gmail": "mixed"}.get(dataset, "")


# ---------------------------------------------------------------------------
# dataset_summary.json
# ---------------------------------------------------------------------------


def write_dataset_summary(datasets, rows, exact, near, unmatched, errors, cfg, path) -> None:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_dataset[r["dataset"]].append(r)

    summary: dict[str, Any] = {}
    for ds in datasets:
        drows = by_dataset.get(ds.name, [])
        errs = [e for e in errors if e["dataset"] == ds.name]
        summary[ds.name] = _summarize_one(ds, drows, exact, near, unmatched, errs)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    log.info("Wrote %s", path)


def _summarize_one(ds, drows, exact, near, unmatched, errors) -> dict[str, Any]:
    images = sum(1 for r in drows if r.get("is_image"))
    pdfs = sum(1 for r in drows if r.get("is_pdf"))
    anns = sum(1 for r in drows if r.get("is_annotation"))
    unreadable = sum(1 for r in drows if not r.get("is_readable"))
    empty = sum(1 for r in drows if r.get("is_empty"))
    private = sum(1 for r in drows if r.get("is_private"))
    total_size = sum(int(r["size_bytes"]) for r in drows)
    ext_counts = defaultdict(int)
    for r in drows:
        ext_counts[r["extension"]] += 1

    ds_exact = [g for g in exact if any(r["dataset"] == ds.name for r in g)]
    ds_near = [g for g in near if any(r["dataset"] == ds.name for r in g)]
    ds_unmatched = [u for u in unmatched if u["dataset"].split(":")[0] == ds.name]
    unmatched_imgs = sum(1 for u in ds_unmatched if u["file_type"] == "image")
    unmatched_anns = sum(1 for u in ds_unmatched if u["file_type"] == "annotation")

    internal = sorted({str(Path(r["original_relative_path"]).parts[0])
                       for r in drows if r["original_relative_path"]})

    issues: list[str] = []
    if unreadable:
        issues.append(f"{unreadable} unreadable/invalid files")
    if empty:
        issues.append(f"{empty} empty files")
    if ds_exact:
        issues.append(f"{len(ds_exact)} exact duplicate groups")
    if ds_near:
        issues.append(f"{len(ds_near)} near-duplicate groups")
    if unmatched_imgs or unmatched_anns:
        issues.append(f"{unmatched_imgs} unmatched images, {unmatched_anns} unmatched annotations")
    # Flag the bundled pretrained model in SROIE.
    if ds.name == "sroie":
        model_files = [r for r in drows if r["extension"] == ".bin"]
        if model_files:
            issues.append(f"{len(model_files)} pretrained-model artifact(s) preserved (out of scope)")

    return {
        "root_path": str(ds.current_path),
        "target_path": str(ds.target_path),
        "identification_confidence": ds.confidence,
        "evidence": ds.evidence,
        "total_files": len(drows),
        "total_size_bytes": total_size,
        "images": images,
        "pdfs": pdfs,
        "annotations": anns,
        "json_count": sum(1 for r in drows if r["extension"] == ".json"),
        "csv_count": sum(1 for r in drows if r["extension"] == ".csv"),
        "txt_count": sum(1 for r in drows if r["extension"] == ".txt"),
        "xml_count": sum(1 for r in drows if r["extension"] == ".xml"),
        "unreadable_files": unreadable,
        "empty_files": empty,
        "exact_duplicate_groups": len(ds_exact),
        "near_duplicate_groups": len(ds_near),
        "unmatched_images": unmatched_imgs,
        "unmatched_annotations": unmatched_anns,
        "private_file_count": private,
        "document_categories": sorted({r["document_category"] for r in drows}),
        "image_extensions": sorted({r["extension"] for r in drows if r.get("is_image")}),
        "annotation_formats": sorted({r["extension"] for r in drows if r.get("is_annotation")}),
        "extension_counts": dict(ext_counts),
        "internal_structure": internal,
        "issues": issues,
    }


# ---------------------------------------------------------------------------
# Debug aid: limit files per dataset
# ---------------------------------------------------------------------------


class _LimitedDataset:
    """Wrapper that presents a filtered view of a dataset for debugging."""

    def __init__(self, base):
        self.__dict__.update(base.__dict__)

    @classmethod
    def wrap(cls, base, limit):
        obj = cls(base)
        obj._limit = limit
        return obj


if __name__ == "__main__":
    raise SystemExit(main())
