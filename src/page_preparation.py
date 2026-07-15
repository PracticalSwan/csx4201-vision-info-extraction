"""Select document pages and build privacy-safe page manifests.

Public images remain read-only references. Private PDFs are rendered to
anonymized PNG page IDs, while real private paths are confined to an ignored
operational manifest.
"""
from __future__ import annotations

import math
import os
import re
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import fitz
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo

from . import config as cfgmod
from .rotation_common import (
    RotationPipelineError,
    as_bool,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    configuration_hash,
    deterministic_rank,
    ensure_not_raw_output,
    parse_dataset_filter,
    read_csv_rows,
    sha256_file,
    stable_id,
)

PAGE_COLUMNS = [
    "page_id",
    "document_id",
    "source_document_id",
    "source_file_id",
    "dataset",
    "dataset_component",
    "source_type",
    "document_type",
    "language",
    "source_file_path",
    "prepared_image_path",
    "source_file_format",
    "source_page_number",
    "source_page_count",
    "source_width",
    "source_height",
    "prepared_width",
    "prepared_height",
    "prepared_format",
    "materialization_mode",
    "preparation_hash",
    "annotation_availability",
    "annotation_path",
    "private_status",
    "usability_status",
    "sha256",
    "original_dataset_split",
    "logical_document_key",
    "template_family",
    "selection_status",
    "exclusion_reason",
    "notes",
]

PRIVATE_OPERATIONAL_COLUMNS = [
    "page_id",
    "document_id",
    "source_file_id",
    "real_source_path",
    "prepared_image_path",
    "source_page_number",
]

PREPARATION_ERROR_COLUMNS = [
    "timestamp",
    "stage",
    "dataset",
    "page_id",
    "source_file_id",
    "rotation_id",
    "operation",
    "error_type",
    "error_message",
    "action_taken",
]

LANGUAGE = {"sroie": "en", "funsd": "en", "fatura": "tr", "coru": "en", "gmail": "mixed"}
DOCUMENT_TYPE = {"sroie": "receipt", "funsd": "form", "fatura": "invoice", "coru": "receipt", "gmail": "private_document"}
FATURA_TEMPLATE_RE = re.compile(r"^(Template[^_]+)", re.IGNORECASE)


def prepare_page_images(
    cfg: Mapping[str, Any],
    *,
    dry_run: bool = False,
    limit: int = 0,
    datasets: str | list[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Create page manifests and render private PDF pages.

    The limit parameter is a source-file debug cap applied after deterministic
    ordering. It never changes the configured corpus cap used by a normal run.
    """
    started = time.perf_counter()
    root = cfgmod.project_root(cfg)
    metadata_dir = cfgmod.resolve_path(cfg, "metadata")
    reports_dir = cfgmod.resolve_path(cfg, "reports") / "rotation_preparation"
    page_manifest_path = metadata_dir / "page_manifest.csv"
    private_manifest_path = metadata_dir / "private_page_manifest.csv"
    error_path = metadata_dir / "rotation_preparation_errors.csv"
    summary_path = metadata_dir / "rotation_dataset_summary.json"
    report_path = reports_dir / "rotation_preparation_report.md"
    dataset_filter = parse_dataset_filter(datasets)
    preparation_hash = page_preparation_hash(cfg)
    previous_pages = {
        row["page_id"]: row
        for row in read_csv_rows(page_manifest_path)
        if row.get("page_id")
    } if page_manifest_path.is_file() else {}

    inventory_path = metadata_dir / "file_inventory.csv"
    private_inventory_path = metadata_dir / "private_file_inventory.csv"
    if not inventory_path.is_file() or not private_inventory_path.is_file():
        raise FileNotFoundError("organization-stage inventories are required before page preparation")

    public_inventory = read_csv_rows(inventory_path)
    private_inventory = read_csv_rows(private_inventory_path)
    annotations = _annotation_lookup(public_inventory)
    safe_private_by_id = {row["file_id"]: row for row in public_inventory if row.get("dataset") == "gmail"}
    errors: list[dict[str, Any]] = []
    pages: list[dict[str, Any]] = []
    private_operational: list[dict[str, Any]] = []

    public_rows = [
        row for row in public_inventory
        if row.get("dataset") != "gmail" and as_bool(row.get("is_image"))
    ]
    private_rows = [row for row in private_inventory if row.get("dataset") == "gmail" and as_bool(row.get("is_pdf"))]
    if dataset_filter is not None:
        public_rows = [row for row in public_rows if row["dataset"].lower() in dataset_filter]
        if "gmail" not in dataset_filter:
            private_rows = []
    public_rows.sort(key=lambda row: row["file_id"])
    private_rows.sort(key=lambda row: row["file_id"])
    if limit:
        public_rows = public_rows[:limit]
        private_rows = private_rows[:limit]

    for row in public_rows:
        pages.append(_public_page_row(row, cfg, annotations, preparation_hash))

    _apply_public_selection(pages, cfg)
    for page in pages:
        if page["selection_status"] != "selected":
            continue
        source_path = _public_physical_path(root, page["source_file_path"])
        try:
            source_hash = _verified_live_sha256(
                source_path,
                str(page.get("sha256", "")),
                str(page["source_file_id"]),
            )
            page["sha256"] = source_hash
            width, height = _display_dimensions(
                source_path,
                bool(cfg["page_preparation"].get("normalize_exif", True)),
            )
            page["source_width"] = width
            page["source_height"] = height
            if bool(cfg["page_preparation"].get("materialize_existing_images", False)):
                _materialize_public_page(
                    page,
                    source_path,
                    cfg,
                    root,
                    preparation_hash,
                    source_hash,
                    previous_pages.get(page["page_id"]),
                    dry_run=dry_run,
                    force=force,
                )
            else:
                page["prepared_width"] = width
                page["prepared_height"] = height
        except RotationPipelineError:
            raise
        except Exception as exc:
            page["usability_status"] = "excluded"
            page["selection_status"] = "excluded"
            page["exclusion_reason"] = f"image decode failed during preparation: {type(exc).__name__}"
            errors.append(_error(page["dataset"], page["source_file_id"], "inspect_image", exc, "excluded page"))

    for row in private_rows:
        safe_row = safe_private_by_id.get(row["file_id"])
        if safe_row is None:
            errors.append(_error("gmail", row["file_id"], "privacy_join",
                                 RuntimeError("private file has no anonymized public inventory row"), "excluded document"))
            continue
        real_path = _private_physical_path(root, row["current_relative_path"])
        source_hash = _verified_live_sha256(
            real_path,
            str(row.get("sha256", "")),
            str(row["file_id"]),
        )
        try:
            document = fitz.open(real_path)
        except Exception as exc:
            errors.append(_error("gmail", row["file_id"], "open_pdf", exc, "excluded document"))
            continue
        try:
            for page_index in range(document.page_count):
                public_page, operational = _private_page_row(
                    document,
                    page_index,
                    row,
                    safe_row,
                    cfg,
                    root,
                    preparation_hash,
                    source_hash,
                    previous_pages.get(
                        stable_id("gmail_page", row["file_id"], page_index + 1, length=14)
                    ),
                    dry_run=dry_run,
                    force=force,
                )
                pages.append(public_page)
                private_operational.append(operational)
        except Exception as exc:
            errors.append(_error("gmail", row["file_id"], "render_pdf", exc, "recorded; remaining pages skipped"))
        finally:
            document.close()

    pages.sort(key=lambda row: row["page_id"])
    private_operational.sort(key=lambda row: row["page_id"])
    summary = _page_summary(pages, errors, cfg, time.perf_counter() - started)
    if dry_run:
        return {"dry_run": True, "summary": summary, "pages": pages, "errors": errors}

    ensure_not_raw_output(page_manifest_path, root)
    atomic_write_csv(page_manifest_path, pages, PAGE_COLUMNS)
    atomic_write_csv(private_manifest_path, private_operational, PRIVATE_OPERATIONAL_COLUMNS)
    atomic_write_csv(error_path, errors, PREPARATION_ERROR_COLUMNS)
    atomic_write_json(summary_path, summary)
    atomic_write_text(report_path, _preparation_report(summary, cfg))
    return {
        "dry_run": False,
        "page_manifest": str(page_manifest_path),
        "private_manifest": str(private_manifest_path),
        "summary": summary,
        "errors": errors,
    }


def _public_page_row(
    row: Mapping[str, str],
    cfg: Mapping[str, Any],
    annotations: Mapping[tuple[str, str], list[str]],
    preparation_hash: str,
) -> dict[str, Any]:
    dataset = row["dataset"].lower()
    rel = row["current_relative_path"].replace("\\", "/")
    component = _component(dataset, rel)
    original_split = _original_split(dataset, rel)
    logical_key = _logical_document_key(dataset, component, rel)
    document_id = stable_id(f"{dataset}_doc", dataset, logical_key, length=14)
    page_id = stable_id(f"{dataset}_page", row["file_id"], 1, length=14)
    usable, exclusion = _public_usability(row, component, cfg)
    annotation_paths = annotations.get((dataset, row.get("document_id", "")), [])
    template_family = _template_family(dataset, Path(rel).stem)
    return {
        "page_id": page_id,
        "document_id": document_id,
        "source_document_id": row.get("document_id", ""),
        "source_file_id": row["file_id"],
        "dataset": dataset,
        "dataset_component": component,
        "source_type": "public",
        "document_type": DOCUMENT_TYPE.get(dataset, row.get("document_category", "document")),
        "language": LANGUAGE.get(dataset, ""),
        "source_file_path": rel,
        "prepared_image_path": f"data/raw/public/{rel}",
        "source_file_format": row.get("extension", "").lower(),
        "source_page_number": 1,
        "source_page_count": 1,
        "source_width": "",
        "source_height": "",
        "prepared_width": "",
        "prepared_height": "",
        "prepared_format": row.get("extension", "").lstrip(".").lower(),
        "materialization_mode": "referenced_image",
        "preparation_hash": preparation_hash,
        "annotation_availability": "yes" if annotation_paths else "no",
        "annotation_path": annotation_paths[0] if annotation_paths else "",
        "private_status": "public",
        "usability_status": "usable" if usable else "excluded",
        "sha256": row.get("sha256", ""),
        "original_dataset_split": original_split,
        "logical_document_key": logical_key,
        "template_family": template_family,
        "selection_status": "candidate" if usable else "excluded",
        "exclusion_reason": exclusion,
        "notes": "positive rotation angle is counterclockwise",
    }


def _private_page_row(
    document: fitz.Document,
    page_index: int,
    private_row: Mapping[str, str],
    safe_row: Mapping[str, str],
    cfg: Mapping[str, Any],
    root: Path,
    preparation_hash: str,
    source_hash: str,
    previous_page: Mapping[str, str] | None,
    *,
    dry_run: bool,
    force: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_file_id = private_row["file_id"]
    document_id = stable_id("gmail_doc", source_file_id, length=14)
    page_number = page_index + 1
    page_id = stable_id("gmail_page", source_file_id, page_number, length=14)
    output_root = cfgmod.resolve_path(cfg, "private_page_images")
    output_path = output_root / f"{page_id}.png"
    ensure_not_raw_output(output_path, root)
    page = document.load_page(page_index)
    source_width = int(round(page.rect.width))
    source_height = int(round(page.rect.height))
    dpi = int(cfg["page_preparation"].get("pdf_dpi", 200))
    prepared_width = prepared_height = ""
    real_path = _private_physical_path(root, private_row["current_relative_path"])
    provenance = {
        "rotation_pipeline_artifact": "private_pdf_page",
        "preparation_hash": preparation_hash,
        "page_id": page_id,
        "source_sha256": source_hash,
        "source_page_number": str(page_number),
    }

    if (
        output_path.exists()
        and not force
        and _previous_page_matches(
            previous_page,
            output_path,
            preparation_hash,
            source_hash,
            page_number,
            root,
        )
        and _png_artifact_matches(output_path, provenance)
    ):
        with Image.open(output_path) as existing:
            existing.load()
            prepared_width, prepared_height = existing.size
    elif not dry_run:
        pixmap = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
        prepared_width, prepared_height = pixmap.width, pixmap.height
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fd, raw_tmp = tempfile.mkstemp(prefix=f".{page_id}.", suffix=".png", dir=output_path.parent)
        os.close(fd)
        tmp = Path(raw_tmp)
        try:
            rendered = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            rendered.save(tmp, format="PNG", optimize=False, pnginfo=_png_info(provenance))
            with Image.open(tmp) as check:
                check.verify()
            os.replace(tmp, output_path)
        finally:
            tmp.unlink(missing_ok=True)
    else:
        scale = dpi / 72.0
        prepared_width = int(math.ceil(source_width * scale))
        prepared_height = int(math.ceil(source_height * scale))

    safe_source = safe_row["current_relative_path"].replace("\\", "/")
    prepared_rel = output_path.relative_to(root).as_posix()
    public_page = {
        "page_id": page_id,
        "document_id": document_id,
        "source_document_id": private_row.get("document_id", ""),
        "source_file_id": source_file_id,
        "dataset": "gmail",
        "dataset_component": Path(safe_source).parent.name or "private",
        "source_type": "private",
        "document_type": "private_document",
        "language": "mixed",
        "source_file_path": safe_source,
        "prepared_image_path": prepared_rel,
        "source_file_format": ".pdf",
        "source_page_number": page_number,
        "source_page_count": document.page_count,
        "source_width": source_width,
        "source_height": source_height,
        "prepared_width": prepared_width,
        "prepared_height": prepared_height,
        "prepared_format": "png",
        "materialization_mode": "converted_pdf_page",
        "preparation_hash": preparation_hash,
        "annotation_availability": "no",
        "annotation_path": "",
        "private_status": "private",
        "usability_status": "usable",
        "sha256": source_hash,
        "original_dataset_split": "private",
        "logical_document_key": source_file_id,
        "template_family": "",
        "selection_status": "selected",
        "exclusion_reason": "",
        "notes": "private_test only; positive rotation angle is counterclockwise",
    }
    operational = {
        "page_id": page_id,
        "document_id": document_id,
        "source_file_id": source_file_id,
        "real_source_path": str(real_path),
        "prepared_image_path": str(output_path),
        "source_page_number": page_number,
    }
    return public_page, operational


def page_preparation_hash(cfg: Mapping[str, Any]) -> str:
    """Hash every setting that can change a prepared page artifact."""
    return configuration_hash({
        "version": "page-preparation-v2",
        "page_preparation": cfg.get("page_preparation", {}),
    })


def _materialize_public_page(
    page: dict[str, Any],
    source_path: Path,
    cfg: Mapping[str, Any],
    root: Path,
    preparation_hash: str,
    source_hash: str,
    previous_page: Mapping[str, str] | None,
    *,
    dry_run: bool,
    force: bool,
) -> None:
    output_path = cfgmod.resolve_path(cfg, "page_images") / f"{page['page_id']}.png"
    ensure_not_raw_output(output_path, root)
    provenance = {
        "rotation_pipeline_artifact": "public_page",
        "preparation_hash": preparation_hash,
        "page_id": page["page_id"],
        "source_sha256": source_hash,
        "source_page_number": "1",
    }
    reusable = (
        output_path.exists()
        and not force
        and _previous_page_matches(
            previous_page,
            output_path,
            preparation_hash,
            source_hash,
            1,
            root,
        )
        and _png_artifact_matches(output_path, provenance)
    )
    if reusable:
        with Image.open(output_path) as existing:
            existing.load()
            prepared_size = existing.size
    else:
        with Image.open(source_path) as opened:
            image = opened
            if bool(cfg["page_preparation"].get("normalize_exif", True)):
                image = ImageOps.exif_transpose(image)
            converted = image.convert("RGB")
            prepared_size = converted.size
            if not dry_run:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                fd, raw_tmp = tempfile.mkstemp(
                    prefix=f".{page['page_id']}.", suffix=".png", dir=output_path.parent
                )
                os.close(fd)
                tmp = Path(raw_tmp)
                try:
                    converted.save(
                        tmp,
                        format="PNG",
                        optimize=False,
                        pnginfo=_png_info(provenance),
                    )
                    with Image.open(tmp) as check:
                        check.verify()
                    os.replace(tmp, output_path)
                finally:
                    tmp.unlink(missing_ok=True)
    page["prepared_image_path"] = output_path.relative_to(root).as_posix()
    page["prepared_width"], page["prepared_height"] = prepared_size
    page["prepared_format"] = "png"
    page["materialization_mode"] = "converted_image"
    page["preparation_hash"] = preparation_hash


def _verified_live_sha256(path: Path, audited_hash: str, source_file_id: str) -> str:
    """Return the live source hash and stop if the audit inventory is stale."""
    live_hash = sha256_file(path)
    if audited_hash and audited_hash != live_hash:
        raise RotationPipelineError(
            f"live source hash differs from audited inventory for source_file_id={source_file_id}"
        )
    return live_hash


def _previous_page_matches(
    previous_page: Mapping[str, str] | None,
    output_path: Path,
    preparation_hash: str,
    source_hash: str,
    page_number: int,
    root: Path,
) -> bool:
    if not previous_page:
        return False
    try:
        previous_output = (root / previous_page.get("prepared_image_path", "")).resolve()
        return (
            previous_output == output_path.resolve()
            and previous_page.get("preparation_hash") == preparation_hash
            and previous_page.get("sha256") == source_hash
            and int(previous_page.get("source_page_number", "0")) == page_number
        )
    except (OSError, TypeError, ValueError):
        return False


def _png_info(values: Mapping[str, Any]) -> PngInfo:
    info = PngInfo()
    for key, value in values.items():
        info.add_text(str(key), str(value))
    return info


def _png_artifact_matches(path: Path, expected: Mapping[str, Any]) -> bool:
    try:
        with Image.open(path) as image:
            image.load()
            if image.format != "PNG" or image.mode != "RGB" or image.width <= 0 or image.height <= 0:
                return False
            return all(str(image.info.get(key, "")) == str(value) for key, value in expected.items())
    except (OSError, ValueError):
        return False


def _apply_public_selection(pages: list[dict[str, Any]], cfg: Mapping[str, Any]) -> None:
    seed = int(cfg.get("rotation_splits", {}).get("seed", 42))
    cap = int(cfg.get("page_selection", {}).get("max_pages_per_public_dataset", 0))
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for page in pages:
        if page["source_type"] == "public" and page["usability_status"] == "usable":
            by_dataset[page["dataset"]].append(page)

    selected_ids: set[str] = set()
    for dataset, rows in sorted(by_dataset.items()):
        if cap <= 0 or len(rows) <= cap:
            selected_ids.update(row["page_id"] for row in rows)
            continue
        if dataset == "fatura":
            selected = _round_robin(rows, cap, lambda row: row["template_family"] or "unknown", seed)
        elif dataset == "coru":
            selected = []
            components = defaultdict(list)
            for row in rows:
                components[row["dataset_component"]].append(row)
            component_names = sorted(components)
            base = cap // max(1, len(component_names))
            remainder = cap % max(1, len(component_names))
            for index, component in enumerate(component_names):
                quota = base + (1 if index < remainder else 0)
                selected.extend(_proportional_select(
                    components[component],
                    quota,
                    lambda row: row["original_dataset_split"] or "unknown",
                    seed,
                ))
        else:
            selected = _proportional_select(
                rows,
                cap,
                lambda row: row["original_dataset_split"] or "unknown",
                seed,
            )
        selected_ids.update(row["page_id"] for row in selected[:cap])

    for page in pages:
        if page["source_type"] != "public" or page["usability_status"] != "usable":
            continue
        if page["page_id"] in selected_ids:
            page["selection_status"] = "selected"
        else:
            page["selection_status"] = "excluded"
            page["exclusion_reason"] = "deterministic bounded-corpus cap"


def _proportional_select(rows, cap, stratum_key, seed):
    if cap >= len(rows):
        return list(rows)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(stratum_key(row))].append(row)
    quotas: dict[str, int] = {}
    fractions: list[tuple[float, str]] = []
    for key, group in grouped.items():
        exact = cap * len(group) / len(rows)
        quotas[key] = min(len(group), int(math.floor(exact)))
        fractions.append((exact - quotas[key], key))
    remaining = cap - sum(quotas.values())
    for _, key in sorted(fractions, key=lambda item: (-item[0], item[1])):
        if remaining <= 0:
            break
        if quotas[key] < len(grouped[key]):
            quotas[key] += 1
            remaining -= 1
    selected: list[dict[str, Any]] = []
    for key in sorted(grouped):
        ordered = sorted(grouped[key], key=lambda row: deterministic_rank(row["page_id"], seed))
        selected.extend(ordered[:quotas[key]])
    if len(selected) < cap:
        selected_ids = {row["page_id"] for row in selected}
        leftovers = sorted(
            (row for row in rows if row["page_id"] not in selected_ids),
            key=lambda row: deterministic_rank(row["page_id"], seed),
        )
        selected.extend(leftovers[:cap - len(selected)])
    return selected


def _round_robin(rows, cap, stratum_key, seed):
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(stratum_key(row))].append(row)
    for key in grouped:
        grouped[key].sort(key=lambda row: deterministic_rank(row["page_id"], seed))
    selected: list[dict[str, Any]] = []
    depth = 0
    keys = sorted(grouped)
    while len(selected) < cap:
        progressed = False
        for key in keys:
            if depth < len(grouped[key]):
                selected.append(grouped[key][depth])
                progressed = True
                if len(selected) == cap:
                    break
        if not progressed:
            break
        depth += 1
    return selected


def _annotation_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[str]]:
    lookup: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in rows:
        if row.get("dataset") == "gmail" or not as_bool(row.get("is_annotation")):
            continue
        lookup[(row["dataset"], row.get("document_id", ""))].append(
            row["current_relative_path"].replace("\\", "/")
        )
    for paths in lookup.values():
        paths.sort()
    return lookup


def _public_usability(row: Mapping[str, str], component: str, cfg: Mapping[str, Any]) -> tuple[bool, str]:
    if not as_bool(row.get("is_readable")) or as_bool(row.get("is_empty")):
        return False, "source image is unreadable, invalid, or empty"
    rel = row["current_relative_path"].replace("\\", "/").lower()
    if "__macosx" in rel or Path(rel).name.startswith("._"):
        return False, "macOS metadata artifact"
    supported = {str(ext).lower() for ext in cfg["page_preparation"].get("supported_image_extensions", [])}
    if row.get("extension", "").lower() not in supported:
        return False, "unsupported image format"
    if row["dataset"].lower() == "coru":
        allowed = set(cfg["page_selection"].get("coru_components", []))
        if component not in allowed:
            reason = cfg["page_selection"].get("exclude_coru_components", {}).get(
                component, "component is not an approved full-document image source"
            )
            return False, reason
    return True, ""


def _component(dataset: str, rel: str) -> str:
    parts = rel.replace("\\", "/").split("/")
    if dataset == "coru" and len(parts) > 1:
        return parts[1]
    if len(parts) > 1:
        return parts[1]
    return dataset


def _original_split(dataset: str, rel: str) -> str:
    parts = [part.lower() for part in rel.replace("\\", "/").split("/")]
    if dataset == "funsd":
        if "training_data" in parts:
            return "train"
        if "testing_data" in parts:
            return "test"
    for split in ("train", "validation", "val", "test", "dev"):
        if split in parts:
            return "validation" if split in {"validation", "val", "dev"} else split
    return "unspecified"


def _logical_document_key(dataset: str, component: str, rel: str) -> str:
    stem = Path(rel).stem
    if dataset == "coru":
        return f"{component}/{stem}"
    return stem


def _template_family(dataset: str, stem: str) -> str:
    if dataset != "fatura":
        return ""
    match = FATURA_TEMPLATE_RE.match(stem)
    return match.group(1).lower() if match else "unknown"


def _public_physical_path(root: Path, safe_rel: str) -> Path:
    rel = safe_rel.replace("\\", "/")
    return root / "data" / "raw" / "public" / Path(rel)


def _private_physical_path(root: Path, real_rel: str) -> Path:
    rel = real_rel.replace("\\", "/")
    return root / "data" / "raw" / "private" / Path(rel)


def _display_dimensions(path: Path, normalize_exif: bool) -> tuple[int, int]:
    with Image.open(path) as image:
        if normalize_exif:
            image = ImageOps.exif_transpose(image)
        return int(image.width), int(image.height)


def _error(dataset: str, source_file_id: str, operation: str, exc: Exception, action: str) -> dict[str, Any]:
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage": "pages",
        "dataset": dataset,
        "page_id": "",
        "source_file_id": source_file_id,
        "rotation_id": "",
        "operation": operation,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "action_taken": action,
    }


def _page_summary(
    pages: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    cfg: Mapping[str, Any],
    elapsed: float,
) -> dict[str, Any]:
    selected = [row for row in pages if row["selection_status"] == "selected"]
    exclusions = Counter(row["exclusion_reason"] for row in pages if row["selection_status"] == "excluded")
    return {
        "stage": "page_preparation",
        "rotation_direction": "counterclockwise",
        "full_profile_scope": cfg["rotation_generation"].get("full_profile_scope"),
        "candidate_page_rows": len(pages),
        "selected_pages": len(selected),
        "selected_by_dataset": dict(Counter(row["dataset"] for row in selected)),
        "selected_by_component": dict(Counter(
            f"{row['dataset']}:{row['dataset_component']}" for row in selected
        )),
        "private_documents": len({row["document_id"] for row in selected if row["private_status"] == "private"}),
        "private_pages": sum(row["private_status"] == "private" for row in selected),
        "excluded_pages": len(pages) - len(selected),
        "exclusion_reasons": {key: value for key, value in exclusions.items() if key},
        "preparation_errors": len(errors),
        "elapsed_seconds": elapsed,
        "public_existing_images_materialized": bool(cfg["page_preparation"].get("materialize_existing_images", False)),
        "pdf_dpi": int(cfg["page_preparation"].get("pdf_dpi", 200)),
    }


def _preparation_report(summary: Mapping[str, Any], cfg: Mapping[str, Any]) -> str:
    selected = summary.get("selected_by_dataset", {})
    lines = [
        "# Rotation Preparation Report",
        "",
        "Positive rotation angles are counterclockwise. Zones use exact half-open intervals:",
        "Zone 1 [0,90), Zone 2 [90,180), Zone 3 [180,270), Zone 4 [270,360).",
        "",
        "## Page preparation",
        "",
        f"- Scope: {summary.get('full_profile_scope')}",
        f"- Selected pages: {summary.get('selected_pages')}",
        f"- Private PDF pages rendered at {summary.get('pdf_dpi')} DPI: {summary.get('private_pages')}",
        f"- Public images are read-only references: {not summary.get('public_existing_images_materialized')}",
        f"- Preparation errors: {summary.get('preparation_errors')}",
        "",
        "## Selected pages by dataset",
        "",
    ]
    for dataset, count in sorted(selected.items()):
        lines.append(f"- {dataset}: {count}")
    lines.extend([
        "",
        "## Resource adaptation",
        "",
        "The full profile applies every required angle to the deterministic selected corpus.",
        "It is not described as a full-corpus run. Full materialization remains subject to",
        f"the smoke-derived disk gate and a {cfg['runtime'].get('minimum_free_space_gb', 10)} GiB reserve.",
        "",
        "Private filenames and source paths are absent from this report and all public metadata.",
    ])
    return "\n".join(lines) + "\n"
