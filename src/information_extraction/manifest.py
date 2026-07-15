"""Build normalized public annotations and leakage-safe IE manifests."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from src import config as cfgmod
from src.information_extraction.annotations import (
    AnnotationConversionError,
    normalize_public_page,
)
from src.rotation_common import atomic_write_csv, atomic_write_json, atomic_write_text, read_csv_rows

MANIFEST_COLUMNS = (
    "document_id", "page_id", "dataset", "dataset_component", "document_type", "language",
    "image_path", "annotation_path", "normalized_annotation_path", "project_split",
    "duplicate_group_id", "has_text_annotation", "has_entity_annotation",
    "has_relation_annotation", "has_field_annotation", "is_private", "is_usable",
    "exclusion_reason", "sha256", "notes",
)
ERROR_COLUMNS = ("page_id", "dataset", "source_path", "error_type", "message")


def build_information_extraction_manifest(
    cfg: Mapping[str, Any], *, limit: int = 0, force: bool = False
) -> dict[str, Any]:
    """Normalize every supported public annotation and build safe manifests."""
    root = cfgmod.project_root(cfg)
    metadata = cfgmod.resolve_path(cfg, "metadata")
    page_manifest_path = metadata / "page_manifest.csv"
    split_manifest_path = metadata / "split_manifest.csv"
    if not page_manifest_path.is_file():
        raise FileNotFoundError(page_manifest_path)
    page_rows = read_csv_rows(page_manifest_path)
    if limit:
        page_rows = page_rows[:limit]
    existing_splits = {
        row["page_id"]: row for row in read_csv_rows(split_manifest_path)
    } if split_manifest_path.is_file() else {}
    fatura_splits = _load_fatura_splits(cfgmod.resolve_path(cfg, "fatura"))
    normalized_root = _normalized_root(cfg, root)
    normalized_root.mkdir(parents=True, exist_ok=True)
    public_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    dataset_counts: Counter[str] = Counter()

    for page in page_rows:
        dataset = page["dataset"].lower()
        private = page.get("private_status") == "private" or dataset == "gmail"
        split_info = existing_splits.get(page["page_id"], {})
        duplicate_group = (
            split_info.get("split_group_id")
            or split_info.get("exact_duplicate_group")
            or split_info.get("near_duplicate_group")
            or page.get("template_family")
            or page.get("logical_document_key")
            or page["document_id"]
        )
        if private:
            private_row = _base_manifest_row(page, "private_test", duplicate_group, private=True)
            private_row.update({
                "image_path": page.get("prepared_image_path", ""),
                "is_usable": "true" if page.get("usability_status") == "usable" else "false",
                "exclusion_reason": "private_test_only",
                "notes": "operational private path; ignored by Git",
            })
            private_rows.append(private_row)
            safe_row = dict(private_row)
            safe_row["image_path"] = ""
            safe_row["annotation_path"] = ""
            safe_row["normalized_annotation_path"] = ""
            safe_row["duplicate_group_id"] = ""
            safe_row["sha256"] = ""
            safe_row["notes"] = "private test row; paths and source fingerprints intentionally omitted"
            public_rows.append(safe_row)
            counts["private_test_rows"] += 1
            continue

        project_split = _project_split(page, split_info, fatura_splits)
        manifest_row = _base_manifest_row(page, project_split, duplicate_group, private=False)
        exclusion = _preflight_exclusion(page)
        record: dict[str, Any] | None = None
        if not exclusion:
            try:
                record = normalize_public_page(page, root, project_split)
                output_rel = Path("data/processed/normalized_ie_annotations") / dataset / f"{page['page_id']}.json"
                output_path = normalized_root / dataset / f"{page['page_id']}.json"
                if force or not output_path.is_file():
                    atomic_write_json(output_path, record)
                provenance = record["annotation_provenance"]["source_paths"]
                manifest_row.update({
                    "annotation_path": provenance[0] if provenance else "",
                    "normalized_annotation_path": output_rel.as_posix(),
                    "has_text_annotation": _flag(bool(record["tokens"] or record["source_qa"])),
                    "has_entity_annotation": _flag(bool(record["entities"])),
                    "has_relation_annotation": _flag(bool(record["relations"] or record["source_qa"])),
                    "has_field_annotation": _flag(bool(record["canonical_fields"])),
                })
                if record["alignment_status"] == "unsupported":
                    exclusion = "unsupported_annotation_semantics"
                elif not record["tokens"] and not record["source_qa"]:
                    exclusion = "no_usable_text_annotation"
                else:
                    manifest_row["is_usable"] = "true"
                    counts["normalized_pages"] += 1
                    split_counts[project_split] += 1
                    dataset_counts[dataset] += 1
            except (AnnotationConversionError, OSError, ValueError, json.JSONDecodeError) as exc:
                exclusion = _annotation_error_exclusion(exc)
                errors.append({
                    "page_id": page["page_id"],
                    "dataset": dataset,
                    "source_path": Path(page.get("prepared_image_path", "")).as_posix(),
                    "error_type": type(exc).__name__,
                    "message": str(exc)[:1000],
                })
        if exclusion:
            manifest_row["is_usable"] = "false"
            manifest_row["exclusion_reason"] = exclusion
            counts[f"excluded:{exclusion}"] += 1
        public_rows.append(manifest_row)

    manifest_path = metadata / "information_extraction_manifest.csv"
    private_manifest_path = metadata / "private_information_extraction_manifest.csv"
    errors_path = metadata / "annotation_normalization_errors.csv"
    atomic_write_csv(manifest_path, public_rows, MANIFEST_COLUMNS)
    atomic_write_csv(private_manifest_path, private_rows, MANIFEST_COLUMNS)
    atomic_write_csv(errors_path, errors, ERROR_COLUMNS)
    summary = {
        "schema_version": "1.0",
        "source_page_count": len(page_rows),
        "public_manifest_rows": len(public_rows),
        "private_operational_rows": len(private_rows),
        "normalized_pages": counts["normalized_pages"],
        "normalization_error_count": len(errors),
        "gmail_fit_rows": 0,
        "counts_by_dataset": dict(sorted(dataset_counts.items())),
        "usable_counts_by_split": dict(sorted(split_counts.items())),
        "exclusion_counts": {
            key.removeprefix("excluded:"): value
            for key, value in sorted(counts.items()) if key.startswith("excluded:")
        },
        "normalized_root": str(normalized_root),
    }
    atomic_write_json(metadata / "information_extraction_manifest_summary.json", summary)
    atomic_write_text(metadata / "annotation_mapping_report.md", _mapping_report(summary))
    return summary


def _normalized_root(cfg: Mapping[str, Any], project_root: Path) -> Path:
    configured = cfg.get("information_extraction", {}).get(
        "normalized_annotations", "data/processed/normalized_ie_annotations"
    )
    path = Path(str(configured))
    return path if path.is_absolute() else project_root / path


def _base_manifest_row(
    page: Mapping[str, str], project_split: str, duplicate_group: str, *, private: bool
) -> dict[str, Any]:
    return {
        "document_id": page["document_id"], "page_id": page["page_id"],
        "dataset": page["dataset"], "dataset_component": page.get("dataset_component", ""),
        "document_type": page.get("document_type", "unknown"), "language": page.get("language", "unknown"),
        "image_path": "" if private else Path(page.get("prepared_image_path", "")).as_posix(),
        "annotation_path": "", "normalized_annotation_path": "", "project_split": project_split,
        "duplicate_group_id": duplicate_group, "has_text_annotation": "false",
        "has_entity_annotation": "false", "has_relation_annotation": "false",
        "has_field_annotation": "false", "is_private": _flag(private), "is_usable": "false",
        "exclusion_reason": "", "sha256": page.get("sha256", ""), "notes": "",
    }


def _preflight_exclusion(page: Mapping[str, str]) -> str:
    if page.get("usability_status") == "excluded":
        return page.get("exclusion_reason") or "source_page_excluded"
    if page["dataset"] == "coru" and page.get("dataset_component") in {
        "OCR Dataset", "Item Information Extraction"
    }:
        return "unsupported_non_document_component"
    if page.get("annotation_availability") != "yes":
        return "missing_source_annotation"
    return ""


def _project_split(
    page: Mapping[str, str], split_info: Mapping[str, str], fatura_splits: Mapping[str, str]
) -> str:
    existing = split_info.get("project_split")
    if existing in {"train", "validation", "test"}:
        return existing
    dataset = page["dataset"]
    native = (page.get("original_dataset_split") or "").lower()
    if dataset == "fatura":
        native = fatura_splits.get(Path(page["prepared_image_path"]).stem, "train")
    if native in {"test", "testing", "testing_data"}:
        return "test"
    if native in {"validation", "val", "dev", "development"}:
        return "validation"
    group = page.get("template_family") or page.get("logical_document_key") or page["document_id"]
    rank = int(hashlib.sha256(f"42|{group}".encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "validation" if rank < 0.15 else "train"


def _load_fatura_splits(fatura_root: Path) -> dict[str, str]:
    base = fatura_root / "invoices_dataset_final"
    result: dict[str, str] = {}
    for filename, split in (
        ("strat1_train.csv", "train"), ("strat1_dev.csv", "validation"), ("strat1_test.csv", "test")
    ):
        path = base / filename
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                image_name = row.get("img_path") or row.get("image") or ""
                if image_name:
                    result[Path(image_name).stem] = split
    return result


def _annotation_error_exclusion(exc: Exception) -> str:
    if isinstance(exc, json.JSONDecodeError):
        return "malformed_source_annotation"
    if isinstance(exc, AnnotationConversionError) and "annotation missing:" in str(exc).lower():
        return "missing_source_annotation"
    return "annotation_conversion_error"


def _flag(value: bool) -> str:
    return "true" if value else "false"


def _mapping_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Public annotation mapping report",
        "",
        "The adapters preserve raw annotations unchanged and write normalized JSON to the configured generated-data root.",
        "",
        "## Mapping",
        "",
        "- SROIE polygon OCR plus entity JSON: tokens are retained; company, address, date, and total become canonical fields.",
        "- FUNSD form JSON: header, question, answer, and other labels map to the universal entity set; links become relations.",
        "- FATURA LayoutLM-HF plus original JSON: words and boxes are retained; non-background field tags map to VALUE and table tags to TABLE_CELL.",
        "- CORU receipt QA: full-document question-answer supervision is retained for later OCR alignment. KIE YOLO regions are excluded from supervised fitting because no verified text/class map is present.",
        "- CORU OCR line crops and CSV-only item records remain excluded because they are not full-document image examples.",
        "",
        "## Counts",
        "",
        f"- Source pages inspected: {summary['source_page_count']}",
        f"- Normalized usable pages: {summary['normalized_pages']}",
        f"- Conversion errors: {summary['normalization_error_count']}",
        f"- Private Gmail fit rows: {summary['gmail_fit_rows']}",
        f"- Usable by dataset: `{json.dumps(summary['counts_by_dataset'], sort_keys=True)}`",
        f"- Usable by split: `{json.dumps(summary['usable_counts_by_split'], sort_keys=True)}`",
        f"- Exclusions: `{json.dumps(summary['exclusion_counts'], sort_keys=True)}`",
        "",
        "## Known alignment boundary",
        "",
        "CORU QA answers have no source token polygons. They remain valid field/relation supervision only after local OCR alignment passes the configured coverage gate. They are not silently treated as token labels.",
    ]
    return "\n".join(lines) + "\n"
