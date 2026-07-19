"""Leakage-safe splitting, rotation generation, integrity baselines, and checks."""
from __future__ import annotations

import math
import os
import re
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo

from . import config as cfgmod
from .page_preparation import PREPARATION_ERROR_COLUMNS, page_preparation_hash
from .rotation_common import (
    BOUNDARY_ANGLES,
    LeakageError,
    ROTATION_DIRECTION,
    RotationPipelineError,
    UnionFind,
    as_bool,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    configuration_hash,
    deterministic_rank,
    disk_free_bytes,
    disk_total_bytes,
    get_rotation_zone,
    human_bytes,
    normalize_angle,
    parse_dataset_filter,
    read_csv_rows,
    rotation_filename,
    sha256_file,
    stable_id,
)

SPLIT_COLUMNS = [
    "page_id",
    "document_id",
    "source_file_id",
    "dataset",
    "dataset_component",
    "document_type",
    "original_dataset_split",
    "project_split",
    "split_group_id",
    "exact_duplicate_group",
    "near_duplicate_group",
    "template_family",
    "private_status",
    "prepared_image_path",
    "selection_status",
    "exclusion_reason",
]

ROTATION_COLUMNS = [
    "rotation_id",
    "document_id",
    "page_id",
    "dataset",
    "dataset_component",
    "document_type",
    "project_split",
    "source_image_path",
    "rotated_image_path",
    "rotation_angle",
    "normalized_angle",
    "rotation_zone",
    "rotation_direction",
    "source_width",
    "source_height",
    "output_width",
    "output_height",
    "background_fill",
    "interpolation",
    "private_status",
    "generation_profile",
    "configuration_hash",
    "generation_status",
    "error_message",
]

PROJECT_SPLITS = ("train", "validation", "test")
PRIVATE_SPLIT = "private_test"
ANGLE_TOKEN_RE = re.compile(r"_angle_(\d{3}|[0-9p]+)_zone_([1-4])\.png$", re.IGNORECASE)


def record_raw_baseline(cfg: Mapping[str, Any], *, force: bool = False) -> dict[str, Any]:
    """Record live raw counts, size, and deterministic sampled hashes."""
    root = cfgmod.project_root(cfg)
    output = cfgmod.resolve_path(cfg, "reports") / "verification" / "raw_baseline.json"
    if output.exists() and not force:
        return _load_json(output)
    metadata = cfgmod.resolve_path(cfg, "metadata")
    public_rows = read_csv_rows(metadata / "file_inventory.csv")
    private_rows = read_csv_rows(metadata / "private_file_inventory.csv")
    real_private = {row["file_id"]: row for row in private_rows}
    seed = int(cfg.get("runtime", {}).get("random_seed", 42))
    samples: list[dict[str, Any]] = []
    by_dataset: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in public_rows:
        by_dataset[row["dataset"]].append(row)
    for dataset, rows in sorted(by_dataset.items()):
        ordered = sorted(rows, key=lambda row: deterministic_rank(row["file_id"], seed))
        for row in ordered[:20]:
            if dataset == "gmail":
                real = real_private.get(row["file_id"])
                if real is None:
                    raise RotationPipelineError(f"missing private inventory row for {row['file_id']}")
                physical = root / "data" / "raw" / "private" / Path(real["current_relative_path"])
                safe_path = "<private>"
            else:
                physical = root / "data" / "raw" / "public" / Path(row["current_relative_path"])
                safe_path = row["current_relative_path"].replace("\\", "/")
            live_sha = sha256_file(physical)
            recorded_sha = row.get("sha256", "")
            if recorded_sha and live_sha != recorded_sha:
                raise RotationPipelineError(f"raw hash differs from inventory for {row['file_id']}")
            samples.append({
                "file_id": row["file_id"],
                "dataset": dataset,
                "private": dataset == "gmail",
                "safe_path": safe_path,
                "sha256": live_sha,
            })
    count, size = _raw_count_and_size(root)
    payload = {
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "raw_file_count": count,
        "raw_size_bytes": size,
        "raw_size_human": human_bytes(size),
        "sample_count": len(samples),
        "samples": samples,
    }
    atomic_write_json(output, payload)
    return payload


def verify_raw_baseline(cfg: Mapping[str, Any]) -> dict[str, Any]:
    root = cfgmod.project_root(cfg)
    baseline_path = cfgmod.resolve_path(cfg, "reports") / "verification" / "raw_baseline.json"
    if not baseline_path.is_file():
        return {"passed": False, "detail": "raw baseline is missing"}
    baseline = _load_json(baseline_path)
    metadata = cfgmod.resolve_path(cfg, "metadata")
    public_inventory = metadata / "file_inventory.csv"
    private_inventory = metadata / "private_file_inventory.csv"
    missing = [path.name for path in (public_inventory, private_inventory) if not path.is_file()]
    if missing:
        return {"passed": False, "detail": f"required inventory missing: {', '.join(missing)}"}
    public_rows = {row["file_id"]: row for row in read_csv_rows(public_inventory)}
    private_rows = {row["file_id"]: row for row in read_csv_rows(private_inventory)}
    failures: list[str] = []
    for sample in baseline.get("samples", []):
        file_id = sample["file_id"]
        if sample.get("private"):
            row = private_rows.get(file_id)
            if row is None:
                failures.append(f"missing private sample ID {file_id}")
                continue
            path = root / "data" / "raw" / "private" / Path(row["current_relative_path"])
        else:
            row = public_rows.get(file_id)
            if row is None:
                failures.append(f"missing public sample ID {file_id}")
                continue
            path = root / "data" / "raw" / "public" / Path(row["current_relative_path"])
        if not path.is_file():
            failures.append(f"sample path missing for {file_id}")
        elif sha256_file(path) != sample["sha256"]:
            failures.append(f"sample hash changed for {file_id}")
    count, size = _raw_count_and_size(root)
    if count != int(baseline.get("raw_file_count", -1)):
        failures.append(f"raw file count changed: {count} vs {baseline.get('raw_file_count')}")
    if size != int(baseline.get("raw_size_bytes", -1)):
        failures.append(f"raw byte size changed: {size} vs {baseline.get('raw_size_bytes')}")
    return {
        "passed": not failures,
        "detail": "raw counts, size, and sampled hashes are unchanged" if not failures else "; ".join(failures),
        "raw_file_count": count,
        "raw_size_bytes": size,
        "sample_count": len(baseline.get("samples", [])),
    }


def create_rotation_splits(
    cfg: Mapping[str, Any],
    *,
    dry_run: bool = False,
    seed: int | None = None,
) -> dict[str, Any]:
    metadata = cfgmod.resolve_path(cfg, "metadata")
    page_manifest = metadata / "page_manifest.csv"
    if not page_manifest.is_file():
        raise FileNotFoundError("page_manifest.csv is required before split creation")
    pages = read_csv_rows(page_manifest)
    selected_public = [
        row for row in pages
        if row["selection_status"] == "selected" and row["private_status"] == "public"
    ]
    selected_private = [
        row for row in pages
        if row["selection_status"] == "selected" and row["private_status"] == "private"
    ]
    actual_seed = int(seed if seed is not None else cfg["rotation_splits"].get("seed", 42))
    uf = UnionFind(row["page_id"] for row in selected_public)
    page_by_id = {row["page_id"]: row for row in selected_public}

    _union_same_value(uf, selected_public, "document_id")
    exact_group_by_page = _union_exact_hashes(uf, selected_public)
    near_group_by_page = _union_reported_duplicates(uf, selected_public, metadata / "duplicate_report.csv")
    if cfg["page_selection"].get("group_fatura_template_families", True):
        fatura = [row for row in selected_public if row["dataset"] == "fatura" and row.get("template_family")]
        _union_same_value(uf, fatura, "template_family")
    if cfg["page_selection"].get("group_coru_shared_stems", True):
        coru = [row for row in selected_public if row["dataset"] == "coru"]
        by_stem: dict[str, list[str]] = defaultdict(list)
        for row in coru:
            by_stem[Path(row["logical_document_key"]).name.lower()].append(row["page_id"])
        for ids in by_stem.values():
            _union_ids(uf, ids)

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for page_id, row in page_by_id.items():
        groups[uf.find(page_id)].append(row)
    assignment = _assign_groups_to_splits(groups, cfg, actual_seed)
    split_rows: list[dict[str, Any]] = []
    for row in pages:
        project_split = ""
        split_group_id = ""
        if row["selection_status"] == "selected":
            if row["private_status"] == "private":
                project_split = PRIVATE_SPLIT
                split_group_id = stable_id("splitgrp", row["document_id"], length=14)
            else:
                root_id = uf.find(row["page_id"])
                project_split = assignment[root_id]
                member_ids = sorted(member["page_id"] for member in groups[root_id])
                split_group_id = stable_id("splitgrp", *member_ids, length=14)
        split_rows.append({
            "page_id": row["page_id"],
            "document_id": row["document_id"],
            "source_file_id": row["source_file_id"],
            "dataset": row["dataset"],
            "dataset_component": row["dataset_component"],
            "document_type": row["document_type"],
            "original_dataset_split": row["original_dataset_split"],
            "project_split": project_split,
            "split_group_id": split_group_id,
            "exact_duplicate_group": exact_group_by_page.get(row["page_id"], ""),
            "near_duplicate_group": near_group_by_page.get(row["page_id"], ""),
            "template_family": row.get("template_family", ""),
            "private_status": row["private_status"],
            "prepared_image_path": row["prepared_image_path"],
            "selection_status": row["selection_status"],
            "exclusion_reason": row["exclusion_reason"],
        })
    _validate_split_rows(split_rows)
    summary = _split_summary(split_rows, actual_seed)
    if dry_run:
        return {"dry_run": True, "summary": summary, "rows": split_rows}

    split_root = cfgmod.resolve_path(cfg, "splits")
    split_manifest_path = metadata / "split_manifest.csv"
    atomic_write_csv(split_manifest_path, split_rows, SPLIT_COLUMNS)
    for split in (*PROJECT_SPLITS, PRIVATE_SPLIT):
        atomic_write_csv(
            split_root / f"{split}.csv",
            [row for row in split_rows if row["project_split"] == split],
            SPLIT_COLUMNS,
        )
    summary_path = cfgmod.resolve_path(cfg, "reports") / "rotation_preparation" / "split_summary.json"
    atomic_write_json(summary_path, summary)
    return {"dry_run": False, "summary": summary, "split_manifest": str(split_manifest_path)}


def generate_rotation_data(
    cfg: Mapping[str, Any],
    *,
    profile: str = "full",
    dry_run: bool = False,
    force: bool = False,
    limit: int = 0,
    datasets: str | list[str] | None = None,
    splits: str | list[str] | None = None,
    workers: int | None = None,
) -> dict[str, Any]:
    if profile not in {"smoke", "full"}:
        raise ValueError("profile must be smoke or full")
    started = time.perf_counter()
    root = cfgmod.project_root(cfg)
    metadata = cfgmod.resolve_path(cfg, "metadata")
    split_manifest_path = metadata / "split_manifest.csv"
    page_manifest_path = metadata / "page_manifest.csv"
    if not split_manifest_path.is_file() or not page_manifest_path.is_file():
        raise FileNotFoundError("page and split manifests are required before rotation generation")
    split_rows = read_csv_rows(split_manifest_path)
    page_rows = {row["page_id"]: row for row in read_csv_rows(page_manifest_path)}
    dataset_filter = parse_dataset_filter(datasets)
    split_filter = parse_dataset_filter(splits)
    selected = [
        row for row in split_rows
        if row["project_split"]
        and row["selection_status"] == "selected"
        and (dataset_filter is None or row["dataset"].lower() in dataset_filter)
        and (split_filter is None or row["project_split"].lower() in split_filter)
    ]
    if profile == "smoke":
        selected = _smoke_pages(selected, cfg)
    selected.sort(key=lambda row: row["page_id"])
    if limit:
        selected = selected[:limit]
    expected_count = sum(len(_angles_for_split(cfg, profile, row["project_split"])) for row in selected)
    estimate = _resource_estimate(cfg, profile, expected_count)
    unbounded_estimate = estimate_full_corpus_capacity(cfg) if profile == "full" else None
    if profile == "full" and not estimate["safe_to_run"]:
        raise RotationPipelineError(estimate["detail"])
    if dry_run:
        return {
            "dry_run": True,
            "profile": profile,
            "selected_pages": len(selected),
            "expected_rotations": expected_count,
            "resource_estimate": estimate,
            "unbounded_full_corpus_estimate": unbounded_estimate,
        }

    output_root = cfgmod.resolve_path(cfg, "rotated_images") / profile
    config_hash = rotation_configuration_hash(cfg, profile)
    max_workers = max(1, int(workers or cfg.get("runtime", {}).get("workers", 2)))
    rotation_rows: list[dict[str, Any]] = []
    generation_errors: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_page = {
            pool.submit(
                _rotate_page,
                root,
                row,
                page_rows[row["page_id"]],
                cfg,
                profile,
                output_root,
                config_hash,
                force,
            ): row
            for row in selected
        }
        for future in as_completed(future_to_page):
            split_row = future_to_page[future]
            try:
                rows, errors = future.result()
                rotation_rows.extend(rows)
                generation_errors.extend(errors)
            except Exception as exc:
                generation_errors.append(_rotation_error(split_row, "", "rotate_page", exc))
    rotation_rows.sort(key=lambda row: row["rotation_id"])
    generation_errors.sort(key=lambda row: (row.get("page_id", ""), row.get("rotation_id", "")))
    if len(rotation_rows) != expected_count:
        raise RotationPipelineError(
            f"rotation manifest row count {len(rotation_rows)} does not match expected {expected_count}"
        )
    manifest_path = metadata / "rotation_manifest.csv"
    existing_errors = []
    error_path = metadata / "rotation_preparation_errors.csv"
    if error_path.exists():
        existing_errors = read_csv_rows(error_path)
    combined_errors = _dedupe_errors(existing_errors + generation_errors)
    atomic_write_csv(manifest_path, rotation_rows, ROTATION_COLUMNS)
    atomic_write_csv(error_path, combined_errors, PREPARATION_ERROR_COLUMNS)

    elapsed = time.perf_counter() - started
    summary = _rotation_summary(rotation_rows, generation_errors, profile, config_hash, estimate, elapsed)
    if unbounded_estimate is not None:
        summary["unbounded_full_corpus_estimate"] = unbounded_estimate
    summary_path = cfgmod.resolve_path(cfg, "reports") / "rotation_preparation" / f"{profile}_rotation_summary.json"
    atomic_write_json(summary_path, summary)
    if unbounded_estimate is not None:
        atomic_write_json(
            cfgmod.resolve_path(cfg, "reports")
            / "rotation_preparation"
            / "full_corpus_capacity_estimate.json",
            unbounded_estimate,
        )
    if profile == "smoke":
        successful = [
            root / row["rotated_image_path"]
            for row in rotation_rows if row["generation_status"] == "success"
        ]
        average = int(sum(path.stat().st_size for path in successful) / max(1, len(successful)))
        smoke_estimate = {
            "sample_rotation_count": len(successful),
            "average_bytes_per_rotation": average,
            "sample_bytes": sum(path.stat().st_size for path in successful),
            "output_max_dimension": int(cfg["rotation_generation"].get("output_max_dimension", 1024)),
        }
        atomic_write_json(
            cfgmod.resolve_path(cfg, "reports") / "rotation_preparation" / "smoke_resource_estimate.json",
            smoke_estimate,
        )
        # The smoke measurement supplies the empirical bytes-per-rotation value
        # needed for an honest all-usable-corpus capacity estimate.  Persist it
        # here so a smoke-only end-to-end run also has the required disk report.
        smoke_capacity = estimate_full_corpus_capacity(cfg)
        summary["unbounded_full_corpus_estimate"] = smoke_capacity
        atomic_write_json(
            cfgmod.resolve_path(cfg, "reports")
            / "rotation_preparation"
            / "full_corpus_capacity_estimate.json",
            smoke_capacity,
        )
        atomic_write_json(summary_path, summary)
    return {
        "dry_run": False,
        "profile": profile,
        "rotation_manifest": str(manifest_path),
        "summary": summary,
        "errors": generation_errors,
    }


def estimate_full_corpus_capacity(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Estimate the unbounded usable-corpus experiment without materializing it."""
    metadata = cfgmod.resolve_path(cfg, "metadata")
    page_manifest = metadata / "page_manifest.csv"
    if not page_manifest.is_file():
        raise FileNotFoundError(page_manifest)
    pages = read_csv_rows(page_manifest)
    usable_public = [
        row for row in pages
        if row.get("private_status") == "public" and row.get("usability_status") == "usable"
    ]
    usable_private = [
        row for row in pages
        if row.get("private_status") == "private" and row.get("usability_status") == "usable"
    ]
    ratios = {split: float(cfg["rotation_splits"][split]) for split in PROJECT_SPLITS}
    split_pages = _target_counts(len(usable_public), ratios)
    generation = cfg["rotation_generation"]
    train_angle_count = sum(len(generation["train_angles"][f"zone_{zone}"]) for zone in (1, 2, 3, 4))
    boundary_angle_count = len(generation["boundary_angles"])
    private_angle_count = len(generation["private_test_angles"])
    public_rotations = (
        split_pages["train"] * train_angle_count
        + split_pages["validation"] * boundary_angle_count
        + split_pages["test"] * boundary_angle_count
    )
    private_rotations = len(usable_private) * private_angle_count
    expected_rotations = public_rotations + private_rotations
    resource = _resource_estimate(cfg, "full", expected_rotations)
    bounded_cap = int(cfg.get("page_selection", {}).get("max_pages_per_public_dataset", 0))
    return {
        "scope": "all_usable_full_document_pages_before_bounded_selection",
        "usable_public_pages": len(usable_public),
        "usable_private_pages": len(usable_private),
        "estimated_public_pages_by_split": split_pages,
        "train_angles_per_page": train_angle_count,
        "evaluation_angles_per_page": boundary_angle_count,
        "private_angles_per_page": private_angle_count,
        "expected_public_rotations": public_rotations,
        "expected_private_rotations": private_rotations,
        "expected_total_rotations": expected_rotations,
        "resource_estimate": resource,
        "safe_on_current_disk": bool(resource["safe_to_run"]),
        "active_reduced_configuration": {
            "max_pages_per_public_dataset": bounded_cap,
            "reason": "preserve the configured free-space reserve on the current machine",
        },
    }


def verify_rotation_data(
    cfg: Mapping[str, Any],
    *,
    profile: str | None = None,
    require_model_artifacts: bool = False,
    require_portable_artifacts: bool = False,
) -> dict[str, Any]:
    root = cfgmod.project_root(cfg)
    metadata = cfgmod.resolve_path(cfg, "metadata")
    checks: list[dict[str, Any]] = []
    raw = verify_raw_baseline(cfg)
    checks.append({"name": "raw-integrity", "passed": raw["passed"], "detail": raw["detail"]})
    page_path = metadata / "page_manifest.csv"
    split_path = metadata / "split_manifest.csv"
    rotation_path = metadata / "rotation_manifest.csv"
    checks.append({"name": "page-manifest-exists", "passed": page_path.is_file(), "detail": str(page_path)})
    checks.append({"name": "split-manifest-exists", "passed": split_path.is_file(), "detail": str(split_path)})
    checks.append({"name": "rotation-manifest-exists", "passed": rotation_path.is_file(), "detail": str(rotation_path)})
    if not all(path.is_file() for path in (page_path, split_path, rotation_path)):
        return _write_verification(cfg, checks, profile or "unknown")

    pages = read_csv_rows(page_path)
    splits = read_csv_rows(split_path)
    rotations = read_csv_rows(rotation_path)
    private_page_path = metadata / "private_page_manifest.csv"
    private_pages = {
        row["page_id"]: row
        for row in read_csv_rows(private_page_path)
        if row.get("page_id")
    } if private_page_path.is_file() else {}
    active_profile = profile or (rotations[0]["generation_profile"] if rotations else "unknown")
    selected_pages = [row for row in pages if row["selection_status"] == "selected"]
    bad_pages = []
    bad_page_provenance = []
    expected_preparation_hash = page_preparation_hash(cfg)
    live_page_source_hashes: dict[str, str] = {}
    for row in selected_pages:
        path = root / row["prepared_image_path"]
        if not path.is_file():
            bad_pages.append(row["page_id"])
            continue
        try:
            materialization_mode = row.get("materialization_mode", "")
            if materialization_mode == "converted_pdf_page":
                operational = private_pages.get(row["page_id"])
                if not operational or not operational.get("real_source_path"):
                    raise ValueError("private source provenance is unavailable")
                source_path = Path(operational["real_source_path"])
            elif materialization_mode == "converted_image":
                source_path = root / "data" / "raw" / "public" / Path(
                    row["source_file_path"].replace("\\", "/")
                )
            else:
                source_path = path
            source_key = str(source_path.resolve())
            if source_key not in live_page_source_hashes:
                live_page_source_hashes[source_key] = sha256_file(source_path)
            live_source_hash = live_page_source_hashes[source_key]
            with Image.open(path) as image:
                if image.width <= 0 or image.height <= 0:
                    bad_pages.append(row["page_id"])
                if row.get("sha256") != live_source_hash:
                    bad_page_provenance.append(row["page_id"])
                elif row.get("preparation_hash") != expected_preparation_hash:
                    bad_page_provenance.append(row["page_id"])
                elif materialization_mode in {"converted_pdf_page", "converted_image"}:
                    expected = {
                        "rotation_pipeline_artifact": (
                            "private_pdf_page"
                            if materialization_mode == "converted_pdf_page"
                            else "public_page"
                        ),
                        "preparation_hash": expected_preparation_hash,
                        "page_id": row["page_id"],
                        "source_sha256": live_source_hash,
                        "source_page_number": row.get("source_page_number", ""),
                    }
                    if image.format != "PNG" or any(
                        str(image.info.get(key, "")) != str(value)
                        for key, value in expected.items()
                    ):
                        bad_page_provenance.append(row["page_id"])
        except Exception:
            bad_pages.append(row["page_id"])
            bad_page_provenance.append(row["page_id"])
    checks.append({
        "name": "selected-page-paths-readable",
        "passed": not bad_pages,
        "detail": f"{len(selected_pages) - len(bad_pages)}/{len(selected_pages)} readable",
    })
    checks.append({
        "name": "prepared-page-provenance",
        "passed": not bad_page_provenance,
        "detail": (
            f"{len(selected_pages) - len(set(bad_page_provenance))}/{len(selected_pages)} provenance-valid"
        ),
    })
    try:
        _validate_split_rows(splits)
        checks.append({"name": "split-leakage", "passed": True, "detail": "documents, pages, groups, and privacy are isolated"})
    except Exception as exc:
        checks.append({"name": "split-leakage", "passed": False, "detail": str(exc)})

    duplicate_rotation_ids = len(rotations) - len({row["rotation_id"] for row in rotations})
    checks.append({
        "name": "rotation-ids-unique",
        "passed": duplicate_rotation_ids == 0,
        "detail": f"duplicate IDs={duplicate_rotation_ids}",
    })
    successful = [row for row in rotations if row["generation_status"] == "success"]
    invalid_rows: list[str] = []
    manifest_paths: set[str] = set()
    expected_rotation_hash = rotation_configuration_hash(cfg, active_profile)
    source_hashes: dict[str, str] = {}
    for row in successful:
        rel = row["rotated_image_path"].replace("\\", "/")
        manifest_paths.add(rel)
        path = root / rel
        try:
            angle = float(row["normalized_angle"])
            zone = int(row["rotation_zone"])
            if get_rotation_zone(angle) != zone:
                raise ValueError("zone mismatch")
            if f"/zone_{zone}/" not in f"/{rel}/":
                raise ValueError("zone folder mismatch")
            match = ANGLE_TOKEN_RE.search(path.name)
            if match is None or int(match.group(2)) != zone:
                raise ValueError("filename mismatch")
            if row.get("configuration_hash") != expected_rotation_hash:
                raise ValueError("stale rotation configuration hash")
            source_rel = row["source_image_path"].replace("\\", "/")
            if source_rel not in source_hashes:
                source_hashes[source_rel] = sha256_file(root / source_rel)
            expected_metadata = {
                "rotation_pipeline_artifact": "rotated_page",
                "configuration_hash": row["configuration_hash"],
                "rotation_id": row["rotation_id"],
                "source_sha256": source_hashes[source_rel],
            }
            with Image.open(path) as image:
                if image.mode != "RGB" or image.width <= 0 or image.height <= 0:
                    raise ValueError("invalid image mode or dimensions")
                if image.format != "PNG" or any(
                    str(image.info.get(key, "")) != str(value)
                    for key, value in expected_metadata.items()
                ):
                    raise ValueError("rotation artifact provenance mismatch")
        except Exception:
            invalid_rows.append(row["rotation_id"])
    checks.append({
        "name": "rotation-artifacts-valid",
        "passed": not invalid_rows,
        "detail": f"{len(successful) - len(invalid_rows)}/{len(successful)} valid",
    })
    physical = {
        path.relative_to(root).as_posix()
        for path in (cfgmod.resolve_path(cfg, "rotated_images") / active_profile).rglob("*.png")
    } if (cfgmod.resolve_path(cfg, "rotated_images") / active_profile).exists() else set()
    checks.append({
        "name": "manifest-matches-files",
        "passed": manifest_paths == physical,
        "detail": f"manifest={len(manifest_paths)} physical={len(physical)}",
    })
    zone_counts = Counter(int(row["rotation_zone"]) for row in successful)
    all_zones = all(zone_counts.get(zone, 0) > 0 for zone in (1, 2, 3, 4))
    checks.append({"name": "all-zones-represented", "passed": all_zones, "detail": str(dict(zone_counts))})
    balance_ok = bool(zone_counts) and max(zone_counts.values()) - min(zone_counts.values()) <= max(1, int(0.02 * max(zone_counts.values())))
    checks.append({"name": "zone-balance", "passed": balance_ok, "detail": str(dict(zone_counts))})
    if active_profile == "full":
        boundary_by_split = {
            split: {int(float(row["normalized_angle"])) for row in successful if row["project_split"] == split}
            for split in ("validation", "test")
        }
        boundary_ok = all(set(BOUNDARY_ANGLES) <= angles for angles in boundary_by_split.values())
        checks.append({
            "name": "boundary-angles",
            "passed": boundary_ok,
            "detail": str({key: len(value) for key, value in boundary_by_split.items()}),
        })
    leak_result = _scan_private_name_leaks(cfg)
    checks.append({"name": "private-name-leaks", **leak_result})
    raw_output_hits = [
        path for path in (root / "data" / "raw").rglob("*")
        if path.is_file() and any(part in {"processed", "features", "rotated_images", "models", "reports"} for part in path.parts)
    ]
    checks.append({
        "name": "no-generated-files-under-raw",
        "passed": not raw_output_hits,
        "detail": f"hits={len(raw_output_hits)}",
    })
    if require_model_artifacts or require_portable_artifacts:
        checks.extend(
            _verify_completed_pipeline(
                cfg,
                active_profile,
                require_portable_artifacts=require_portable_artifacts,
            )
        )
    return _write_verification(cfg, checks, active_profile)


def _rotate_page(root, split_row, page_row, cfg, profile, output_root, config_hash, force):
    source_path = root / split_row["prepared_image_path"]
    angles = _angles_for_split(cfg, profile, split_row["project_split"])
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    max_dimension = int(cfg["rotation_generation"].get("output_max_dimension", 1024))
    try:
        source_fingerprint = sha256_file(source_path)
        with Image.open(source_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            original_size = image.size
            image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
            prepared = image.copy()
    except Exception as exc:
        for angle in angles:
            rotation_id = stable_id("rotation", split_row["page_id"], angle, profile, length=16)
            rows.append(_failed_rotation_row(split_row, page_row, angle, profile, config_hash, rotation_id, exc))
            errors.append(_rotation_error(split_row, rotation_id, "load_source", exc))
        return rows, errors

    for angle in angles:
        normalized = normalize_angle(angle)
        zone = get_rotation_zone(normalized)
        rotation_id = stable_id("rotation", split_row["page_id"], normalized, profile, length=16)
        filename = rotation_filename(split_row["page_id"], normalized, zone)
        output_path = output_root / split_row["project_split"] / f"zone_{zone}" / filename
        rel_output = output_path.relative_to(root).as_posix()
        provenance = {
            "rotation_pipeline_artifact": "rotated_page",
            "configuration_hash": config_hash,
            "rotation_id": rotation_id,
            "source_sha256": source_fingerprint,
        }
        try:
            if output_path.exists() and not force and _png_artifact_matches(output_path, provenance):
                with Image.open(output_path) as existing:
                    existing.load()
                    output_size = existing.size
            else:
                rotated = prepared.rotate(
                    normalized,
                    resample=Image.Resampling.BICUBIC,
                    expand=True,
                    fillcolor=(255, 255, 255),
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                fd, raw_tmp = tempfile.mkstemp(prefix=f".{rotation_id}.", suffix=".png", dir=output_path.parent)
                os.close(fd)
                tmp = Path(raw_tmp)
                try:
                    rotated.save(
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
                output_size = rotated.size
            rows.append({
                "rotation_id": rotation_id,
                "document_id": split_row["document_id"],
                "page_id": split_row["page_id"],
                "dataset": split_row["dataset"],
                "dataset_component": split_row["dataset_component"],
                "document_type": split_row["document_type"],
                "project_split": split_row["project_split"],
                "source_image_path": split_row["prepared_image_path"],
                "rotated_image_path": rel_output,
                "rotation_angle": angle,
                "normalized_angle": normalized,
                "rotation_zone": zone,
                "rotation_direction": ROTATION_DIRECTION,
                "source_width": original_size[0],
                "source_height": original_size[1],
                "output_width": output_size[0],
                "output_height": output_size[1],
                "background_fill": "white",
                "interpolation": "bicubic",
                "private_status": split_row["private_status"],
                "generation_profile": profile,
                "configuration_hash": config_hash,
                "generation_status": "success",
                "error_message": "",
            })
        except Exception as exc:
            rows.append(_failed_rotation_row(split_row, page_row, angle, profile, config_hash, rotation_id, exc))
            errors.append(_rotation_error(split_row, rotation_id, "rotate_image", exc))
    return rows, errors


def rotation_configuration_hash(cfg: Mapping[str, Any], profile: str) -> str:
    """Hash every configured choice that changes a rotation artifact."""
    return configuration_hash({
        "version": "rotation-generation-v2",
        "page_selection": cfg.get("page_selection", {}),
        "rotation_splits": cfg.get("rotation_splits", {}),
        "rotation_generation": cfg.get("rotation_generation", {}),
        "profile": profile,
    })


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


def _failed_rotation_row(split_row, page_row, angle, profile, config_hash, rotation_id, exc):
    normalized = normalize_angle(angle)
    return {
        "rotation_id": rotation_id,
        "document_id": split_row["document_id"],
        "page_id": split_row["page_id"],
        "dataset": split_row["dataset"],
        "dataset_component": split_row["dataset_component"],
        "document_type": split_row["document_type"],
        "project_split": split_row["project_split"],
        "source_image_path": split_row["prepared_image_path"],
        "rotated_image_path": "",
        "rotation_angle": angle,
        "normalized_angle": normalized,
        "rotation_zone": get_rotation_zone(normalized),
        "rotation_direction": ROTATION_DIRECTION,
        "source_width": page_row.get("prepared_width", ""),
        "source_height": page_row.get("prepared_height", ""),
        "output_width": "",
        "output_height": "",
        "background_fill": "white",
        "interpolation": "bicubic",
        "private_status": split_row["private_status"],
        "generation_profile": profile,
        "configuration_hash": config_hash,
        "generation_status": "failed",
        "error_message": f"{type(exc).__name__}: {exc}",
    }


def _angles_for_split(cfg, profile, split):
    generation = cfg["rotation_generation"]
    if profile == "smoke":
        mapping = generation["smoke_angles"]
        angles = [angle for key in ("zone_1", "zone_2", "zone_3", "zone_4") for angle in mapping[key]]
    elif split == "train":
        mapping = generation["train_angles"]
        angles = [angle for key in ("zone_1", "zone_2", "zone_3", "zone_4") for angle in mapping[key]]
    elif split in {"validation", "test"}:
        angles = list(generation["boundary_angles"])
    elif split == PRIVATE_SPLIT:
        angles = list(generation["private_test_angles"])
    else:
        raise ValueError(f"unsupported project split: {split}")
    if len(angles) > int(generation.get("max_variants_per_page", 20)):
        raise RotationPipelineError(f"angle profile exceeds max_variants_per_page for {split}")
    return [normalize_angle(angle) for angle in angles]


def _smoke_pages(rows, cfg):
    per_group = int(cfg["rotation_generation"].get("smoke_pages_per_dataset_per_split", 1))
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["project_split"])].append(row)
    selected = []
    seed = int(cfg.get("runtime", {}).get("random_seed", 42))
    for key, group in sorted(grouped.items()):
        selected.extend(sorted(group, key=lambda row: deterministic_rank(row["page_id"], seed))[:per_group])
    return selected


def _resource_estimate(cfg, profile, expected_count):
    root = cfgmod.project_root(cfg)
    free = disk_free_bytes(root)
    total = disk_total_bytes(root)
    reserve = int(float(cfg["runtime"].get("minimum_free_space_gb", 10)) * (1024 ** 3))
    multiplier = float(cfg["runtime"].get("disk_estimate_safety_multiplier", 1.35))
    smoke_path = cfgmod.resolve_path(cfg, "reports") / "rotation_preparation" / "smoke_resource_estimate.json"
    if smoke_path.exists():
        smoke = _load_json(smoke_path)
        average = int(smoke.get("average_bytes_per_rotation", 0))
        source = "empirical_smoke"
    else:
        average = 750_000
        source = "conservative_default"
    feature_dim_estimate = 2000
    image_bytes = int(expected_count * average * multiplier)
    feature_bytes = int(expected_count * feature_dim_estimate * 4 * 2.0)
    required = image_bytes + feature_bytes
    predicted_fraction = (total - free + required) / total if total else 1.0
    fraction_limit = float(cfg["runtime"].get("maximum_disk_usage_fraction", 0.98))
    safe = free - required >= reserve and predicted_fraction <= fraction_limit
    if profile == "smoke":
        safe = free > reserve
    detail = (
        f"estimated new bytes={human_bytes(required)}, free={human_bytes(free)}, "
        f"reserve={human_bytes(reserve)}, source={source}, predicted usage={predicted_fraction:.3f}"
    )
    if not safe:
        detail = "insufficient disk for safe materialization: " + detail
    return {
        "safe_to_run": safe,
        "expected_rotation_count": expected_count,
        "average_bytes_per_rotation": average,
        "estimated_rotation_bytes": image_bytes,
        "estimated_feature_working_bytes": feature_bytes,
        "estimated_total_new_bytes": required,
        "free_bytes_before": free,
        "reserve_bytes": reserve,
        "predicted_disk_usage_fraction": predicted_fraction,
        "estimate_source": source,
        "detail": detail,
    }


def _assign_groups_to_splits(groups, cfg, seed):
    ratios = {split: float(cfg["rotation_splits"][split]) for split in PROJECT_SPLITS}
    if not math.isclose(sum(ratios.values()), 1.0, abs_tol=1e-8):
        raise ValueError("public split ratios must sum to 1")
    totals = Counter()
    for members in groups.values():
        totals.update(row["dataset"] for row in members)
    targets = {
        dataset: _target_counts(count, ratios)
        for dataset, count in totals.items()
    }
    assigned = {dataset: Counter() for dataset in totals}
    result: dict[str, str] = {}
    ordered_groups = sorted(
        groups.items(),
        key=lambda item: deterministic_rank("|".join(sorted(row["page_id"] for row in item[1])), seed),
    )
    for root_id, members in ordered_groups:
        contribution = Counter(row["dataset"] for row in members)
        costs: list[tuple[float, int, str]] = []
        for order, split in enumerate(PROJECT_SPLITS):
            cost = 0.0
            for dataset, amount in contribution.items():
                target = max(1, targets[dataset][split])
                before = assigned[dataset][split]
                after = before + amount
                cost += ((after - target) / target) ** 2 - ((before - target) / target) ** 2
            costs.append((cost, order, split))
        chosen = min(costs)[2]
        result[root_id] = chosen
        for dataset, amount in contribution.items():
            assigned[dataset][chosen] += amount
    return result


def _target_counts(total, ratios):
    exact = {split: total * ratio for split, ratio in ratios.items()}
    counts = {split: int(math.floor(value)) for split, value in exact.items()}
    remaining = total - sum(counts.values())
    for split in sorted(PROJECT_SPLITS, key=lambda name: (-(exact[name] - counts[name]), PROJECT_SPLITS.index(name))):
        if remaining <= 0:
            break
        counts[split] += 1
        remaining -= 1
    return counts


def _union_same_value(uf, rows, column):
    by_value: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        value = row.get(column, "")
        if value:
            by_value[value].append(row["page_id"])
    for ids in by_value.values():
        _union_ids(uf, ids)


def _union_exact_hashes(uf, rows):
    by_hash: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row.get("sha256"):
            by_hash[row["sha256"]].append(row["page_id"])
    mapping = {}
    for sha, ids in by_hash.items():
        if len(ids) > 1:
            _union_ids(uf, ids)
            group = stable_id("exact", sha, length=12)
            mapping.update({page_id: group for page_id in ids})
    return mapping


def _union_reported_duplicates(uf, rows, report_path):
    if not report_path.is_file():
        return {}
    path_to_page = {row["source_file_path"].replace("\\", "/"): row["page_id"] for row in rows}
    groups: dict[str, list[str]] = defaultdict(list)
    for row in read_csv_rows(report_path):
        if row.get("duplicate_type") != "likely_near_duplicate":
            continue
        page_id = path_to_page.get(row["file_path"].replace("\\", "/"))
        if page_id:
            groups[row["duplicate_group_id"]].append(page_id)
    mapping = {}
    for group_id, ids in groups.items():
        if len(ids) > 1:
            _union_ids(uf, ids)
            mapping.update({page_id: group_id for page_id in ids})
    return mapping


def _union_ids(uf, ids):
    if not ids:
        return
    first = ids[0]
    for other in ids[1:]:
        uf.union(first, other)


def _validate_split_rows(rows):
    selected = [row for row in rows if row.get("selection_status") == "selected"]
    page_splits: dict[str, set[str]] = defaultdict(set)
    document_splits: dict[str, set[str]] = defaultdict(set)
    group_splits: dict[str, set[str]] = defaultdict(set)
    exact_group_splits: dict[str, set[str]] = defaultdict(set)
    near_group_splits: dict[str, set[str]] = defaultdict(set)
    for row in selected:
        split = row.get("project_split", "")
        if not split:
            raise LeakageError(f"selected page has no split: {row.get('page_id')}")
        page_splits[row["page_id"]].add(split)
        document_splits[row["document_id"]].add(split)
        if row.get("split_group_id"):
            group_splits[row["split_group_id"]].add(split)
        if row.get("exact_duplicate_group"):
            exact_group_splits[row["exact_duplicate_group"]].add(split)
        if row.get("near_duplicate_group"):
            near_group_splits[row["near_duplicate_group"]].add(split)
        if row.get("private_status") == "private" and split != PRIVATE_SPLIT:
            raise LeakageError("private page entered a public split")
        if row.get("private_status") == "public" and split == PRIVATE_SPLIT:
            raise LeakageError("public page entered private_test")
    for label, mapping in (
        ("page", page_splits),
        ("document", document_splits),
        ("group", group_splits),
        ("exact duplicate group", exact_group_splits),
        ("near duplicate group", near_group_splits),
    ):
        crossed = [key for key, values in mapping.items() if len(values) > 1]
        if crossed:
            raise LeakageError(f"{label} leakage across splits: {crossed[:3]}")


def _split_summary(rows, seed):
    selected = [row for row in rows if row["selection_status"] == "selected"]
    by_split = Counter(row["project_split"] for row in selected)
    by_dataset_split: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in selected:
        by_dataset_split[row["dataset"]][row["project_split"]] += 1
    return {
        "seed": seed,
        "selected_pages": len(selected),
        "counts_by_split": dict(by_split),
        "counts_by_dataset_and_split": {key: dict(value) for key, value in by_dataset_split.items()},
        "split_group_count": len({row["split_group_id"] for row in selected}),
        "document_count": len({row["document_id"] for row in selected}),
        "page_leakage": 0,
        "document_leakage": 0,
        "split_group_leakage": 0,
        "private_public_leakage": 0,
    }


def _rotation_error(split_row, rotation_id, operation, exc):
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage": "rotations",
        "dataset": split_row["dataset"],
        "page_id": split_row["page_id"],
        "source_file_id": split_row["source_file_id"],
        "rotation_id": rotation_id,
        "operation": operation,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "action_taken": "recorded; artifact skipped",
    }


def _rotation_summary(rows, errors, profile, config_hash, estimate, elapsed):
    successful = [row for row in rows if row["generation_status"] == "success"]
    return {
        "stage": "rotation_generation",
        "profile": profile,
        "profile_scope": "smoke" if profile == "smoke" else "bounded_selected_corpus_full_angles",
        "rotation_direction": ROTATION_DIRECTION,
        "configuration_hash": config_hash,
        "expected_rows": len(rows),
        "successful_rows": len(successful),
        "failed_rows": len(rows) - len(successful),
        "error_records": len(errors),
        "counts_by_split": dict(Counter(row["project_split"] for row in successful)),
        "counts_by_zone": dict(Counter(str(row["rotation_zone"]) for row in successful)),
        "counts_by_angle": dict(Counter(str(row["normalized_angle"]) for row in successful)),
        "counts_by_dataset": dict(Counter(row["dataset"] for row in successful)),
        "resource_estimate": estimate,
        "elapsed_seconds": elapsed,
    }


def _dedupe_errors(errors):
    out = {}
    for row in errors:
        key = (
            row.get("stage", ""),
            row.get("source_file_id", ""),
            row.get("rotation_id", ""),
            row.get("operation", ""),
            row.get("error_message", ""),
        )
        out[key] = row
    return list(out.values())


def _raw_count_and_size(root):
    count = 0
    size = 0
    for path in (root / "data" / "raw").rglob("*"):
        if path.is_file():
            count += 1
            size += path.stat().st_size
    return count, size


def _scan_private_name_leaks(cfg):
    root = cfgmod.project_root(cfg)
    metadata = cfgmod.resolve_path(cfg, "metadata")
    private_inventory = metadata / "private_file_inventory.csv"
    if not private_inventory.is_file():
        return {"passed": False, "detail": "private inventory is missing"}
    names = [
        row.get("original_filename", "")
        for row in read_csv_rows(private_inventory)
        if row.get("original_filename", "")
    ]
    hits = []
    public_text_files = []
    scan_bases = (
        metadata,
        cfgmod.resolve_path(cfg, "reports"),
        cfgmod.resolve_path(cfg, "rotation_models"),
        root / "src",
        root / "scripts",
        root / "tests",
        root / "docs",
    )
    text_suffixes = {".csv", ".json", ".md", ".txt", ".py", ".yaml", ".yml", ".toml", ".ini", ".log"}
    for base in scan_bases:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.name.startswith("private_"):
                continue
            if path.suffix.lower() not in text_suffixes:
                continue
            public_text_files.append(path)
    public_text_files.extend(
        path
        for path in root.iterdir()
        if path.is_file() and (path.suffix.lower() in text_suffixes or path.name == ".gitignore")
    )
    folded_names = [name.casefold() for name in names]
    for path in sorted(set(public_text_files)):
        text = path.read_text(encoding="utf-8", errors="replace").casefold()
        for name in folded_names:
            if name and name in text:
                hits.append(f"{path.relative_to(root)} contains a private filename")
                break
    return {
        "passed": not hits,
        "detail": (
            "no private filenames in committable code, tests, docs, config, or public artifacts"
            if not hits
            else "; ".join(hits[:5])
        ),
    }


def _verify_completed_pipeline(
    cfg,
    active_profile,
    *,
    require_portable_artifacts: bool = False,
):
    """Validate feature, preprocessing, K-Means, and exact-angle artifacts.

    This is intentionally separate from the preparation-only verifier so the
    orchestrator can gate feature extraction before any model artifacts exist.
    """
    import hashlib
    import json

    import joblib
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    from src.inference.kmeans_display import VersionNeutralKMeans

    metadata = cfgmod.resolve_path(cfg, "metadata")
    model_root = cfgmod.resolve_path(cfg, "rotation_models")
    report_root = cfgmod.resolve_path(cfg, "reports")
    required = [
        metadata / "feature_manifest.csv",
        metadata / "feature_summary.json",
        metadata / "feature_extraction_errors.csv",
        model_root / "feature_config.json",
        model_root / "scaler.joblib",
        model_root / "preprocessing_summary.json",
        model_root / "kmeans.joblib",
        model_root / "cluster_to_zone.json",
        model_root / "training_summary.json",
        model_root / "angle_estimation_config.json",
        report_root / "kmeans_evaluation" / "metrics.json",
        report_root / "kmeans_evaluation" / "predictions.csv",
        report_root / "kmeans_evaluation" / "confusion_matrix.png",
        report_root / "kmeans_evaluation" / "cluster_zone_count_matrix.csv",
        report_root / "rotation_preparation" / "full_corpus_capacity_estimate.json",
        report_root / "angle_estimation" / "angle_metrics.json",
        report_root / "angle_estimation" / "angle_predictions.csv",
        report_root / "angle_estimation" / "angle_error_by_zone.csv",
        report_root / "angle_estimation" / "angle_error_by_angle.csv",
        report_root / "angle_estimation" / "angle_error_histogram.png",
    ]
    if require_portable_artifacts:
        required.extend(
            [
                model_root / "inference_params.json",
                model_root / "inference_params.npz",
                report_root
                / "rotation"
                / "version_neutral_kmeans_parity.json",
            ]
        )
    missing = [str(path) for path in required if not path.is_file()]
    checks = [{
        "name": "complete-pipeline-artifacts",
        "passed": not missing,
        "detail": "all required feature/model/evaluation artifacts exist" if not missing else f"missing={missing[:5]}",
    }]
    if missing:
        return checks

    try:
        feature_config = _load_json(model_root / "feature_config.json")
        feature_summary = _load_json(metadata / "feature_summary.json")
        preprocessing = _load_json(model_root / "preprocessing_summary.json")
        training = _load_json(model_root / "training_summary.json")
        mapping_payload = _load_json(model_root / "cluster_to_zone.json")
        dimension = int(feature_summary["feature_dimension"])
        cache_root = (
            cfgmod.resolve_path(cfg, "features")
            / feature_config["profile"]
            / feature_config["configuration_hash"]
        )
        expected_splits = {"train", "validation", "test", "private_test"}
        cache_failures = []
        live_rotation_hash = sha256_file(metadata / "rotation_manifest.csv")
        if feature_config.get("rotation_manifest_hash") != live_rotation_hash:
            cache_failures.append("feature configuration has a stale rotation-manifest hash")
        if feature_summary.get("rotation_manifest_hash") != live_rotation_hash:
            cache_failures.append("feature summary has a stale rotation-manifest hash")
        if feature_summary.get("configuration_hash") != feature_config.get("configuration_hash"):
            cache_failures.append("feature summary/configuration hash mismatch")
        raw_values = {}
        transformed_values = {}
        for split in sorted(expected_splits):
            raw_path = cache_root / f"{split}.npz"
            transformed_path = cache_root / f"transformed_{split}.npz"
            if not raw_path.is_file() or not transformed_path.is_file():
                cache_failures.append(f"{split}: cache missing")
                continue
            with np.load(raw_path, allow_pickle=False) as data:
                raw_values[split] = {key: data[key] for key in data.files}
            with np.load(transformed_path, allow_pickle=False) as data:
                transformed_values[split] = {key: data[key] for key in data.files}
            raw = raw_values[split]
            transformed = transformed_values[split]
            expected_count = int(feature_summary["counts_per_split"].get(split, -1))
            if raw["X"].shape != (expected_count, dimension):
                cache_failures.append(f"{split}: raw shape={raw['X'].shape}")
            if transformed["X"].shape[0] != expected_count:
                cache_failures.append(f"{split}: transformed rows={transformed['X'].shape[0]}")
            if not np.isfinite(raw["X"]).all() or not np.isfinite(transformed["X"]).all():
                cache_failures.append(f"{split}: non-finite values")
            if str(raw["configuration_hash"][0]) != feature_config["configuration_hash"]:
                cache_failures.append(f"{split}: feature hash mismatch")
            if str(transformed["preprocessing_hash"][0]) != preprocessing["preprocessing_hash"]:
                cache_failures.append(f"{split}: preprocessing hash mismatch")
            private_flags = raw["private"] != 0
            if split == PRIVATE_SPLIT and not np.all(private_flags):
                cache_failures.append("private_test: public feature row")
            if split != PRIVATE_SPLIT and np.any(private_flags):
                cache_failures.append(f"{split}: private feature row")
        manifest_success = [
            row for row in read_csv_rows(metadata / "feature_manifest.csv")
            if row.get("extraction_status") == "success"
        ]
        if len(manifest_success) != sum(int(value) for value in feature_summary["counts_per_split"].values()):
            cache_failures.append("feature manifest count mismatch")
        if {int(row["feature_dimension"]) for row in manifest_success} != {dimension}:
            cache_failures.append("feature manifest dimension mismatch")
        if feature_config["profile"] != active_profile:
            cache_failures.append("feature profile mismatch")
        checks.append({
            "name": "feature-caches-valid",
            "passed": not cache_failures,
            "detail": f"four splits, dimension={dimension}, finite and hash-matched" if not cache_failures else "; ".join(cache_failures[:5]),
        })

        train = raw_values.get("train")
        transformed_train = transformed_values.get("train")
        provenance_failures = []
        if train is None or transformed_train is None:
            provenance_failures.append("training caches unavailable")
        else:
            digest = hashlib.sha256()
            for value in train["rotation_ids"]:
                digest.update(str(value).encode("utf-8"))
                digest.update(b"\n")
            if preprocessing.get("fit_splits") != ["train"]:
                provenance_failures.append("preprocessing fit scope is not train-only")
            if int(preprocessing.get("fit_private_sample_count", -1)) != 0:
                provenance_failures.append("private features used in preprocessing")
            if int(preprocessing.get("fit_rotation_count", -1)) != len(train["X"]):
                provenance_failures.append("preprocessing fit count mismatch")
            if preprocessing.get("fit_rotation_id_hash") != digest.hexdigest():
                provenance_failures.append("preprocessing fit ID hash mismatch")
            if training.get("fit_splits") != ["train"] or int(training.get("fit_private_sample_count", -1)) != 0:
                provenance_failures.append("K-Means fit scope is not public train-only")
            if training.get("fit_rotation_id_hash") != digest.hexdigest():
                provenance_failures.append("K-Means fit ID hash mismatch")
        checks.append({
            "name": "train-only-fit-provenance",
            "passed": not provenance_failures,
            "detail": "scaler, PCA, and K-Means provenance is public train-only" if not provenance_failures else "; ".join(provenance_failures),
        })

        artifact_failures = []
        artifact_paths = [model_root / "scaler.joblib", model_root / "kmeans.joblib"]
        if preprocessing.get("pca_enabled"):
            artifact_paths.append(model_root / "pca.joblib")
        if any(path.suffix != ".joblib" or path.parent.resolve() != model_root.resolve() for path in artifact_paths):
            artifact_failures.append("unexpected artifact path or suffix")
        scaler = joblib.load(model_root / "scaler.joblib")
        kmeans = joblib.load(model_root / "kmeans.joblib")
        if not isinstance(scaler, StandardScaler) or int(getattr(scaler, "n_features_in_", -1)) != dimension:
            artifact_failures.append("saved scaler type or input dimension mismatch")
        if preprocessing.get("pca_enabled"):
            pca = joblib.load(model_root / "pca.joblib")
            if not isinstance(pca, PCA) or int(getattr(pca, "n_components_", -1)) != int(preprocessing["output_dimension"]):
                artifact_failures.append("saved PCA type or output dimension mismatch")
        if not isinstance(kmeans, KMeans) or int(getattr(kmeans, "n_clusters", -1)) != 4:
            artifact_failures.append("saved model is not K-Means k=4")
        mapping = {int(key): int(value) for key, value in mapping_payload["mapping"].items()}
        if set(mapping) != {0, 1, 2, 3} or set(mapping.values()) != {1, 2, 3, 4}:
            artifact_failures.append("cluster-zone mapping is not bijective")
        if mapping_payload.get("training_hash") != training.get("training_hash"):
            artifact_failures.append("mapping training hash mismatch")
        if transformed_train is not None:
            clusters = kmeans.predict(transformed_train["X"])
            if set(int(value) for value in np.unique(clusters)) != {0, 1, 2, 3}:
                artifact_failures.append("training predictions do not use four clusters")
        if require_portable_artifacts:
            portable = VersionNeutralKMeans(model_root)
            if (
                portable.scaler_mean.shape != (dimension,)
                or portable.pca_components.shape[0]
                != int(preprocessing["output_dimension"])
                or portable.n_clusters != 4
            ):
                artifact_failures.append(
                    "version-neutral inference parameter dimensions mismatch"
                )
            parity = _load_json(
                report_root / "rotation" / "version_neutral_kmeans_parity.json"
            )
            expected_public_rows = sum(
                int(feature_summary["counts_per_split"].get(split, 0))
                for split in ("train", "validation", "test")
            )
            if (
                parity.get("status") != "pass"
                or not parity.get("public_only")
                or int(parity.get("private_rows", -1)) != 0
                or int(parity.get("rows", -1)) != expected_public_rows
                or int(parity.get("cluster_label_mismatches", -1)) != 0
                or parity.get("parameters_sha256")
                != _load_json(model_root / "inference_params.json").get(
                    "parameters_sha256"
                )
            ):
                artifact_failures.append(
                    "version-neutral public inference parity is missing or stale"
                )
        if not preprocessing.get("artifact_reload_verified") or not training.get("artifact_reload_verified"):
            artifact_failures.append("artifact reload verification flag is false")
        checks.append({
            "name": "model-artifacts-compatible",
            "passed": not artifact_failures,
            "detail": (
                (
                    "typed maintenance artifacts and version-neutral inference "
                    "parameters reload; k=4, parity, and one-to-one mapping verified"
                    if require_portable_artifacts
                    else "typed maintenance artifacts reload; k=4 and "
                    "one-to-one mapping verified"
                )
                if not artifact_failures
                else "; ".join(artifact_failures)
            ),
        })

        evaluation_failures = []
        kmeans_metrics = json.loads((report_root / "kmeans_evaluation" / "metrics.json").read_text(encoding="utf-8"))
        angle_metrics = json.loads((report_root / "angle_estimation" / "angle_metrics.json").read_text(encoding="utf-8"))
        if set(kmeans_metrics.get("classification_and_clustering_metrics", {})) != expected_splits:
            evaluation_failures.append("K-Means metrics do not cover all splits")
        private_aggregate = kmeans_metrics.get("private_test_aggregate") or {}
        if int(private_aggregate.get("sample_count", 0)) <= 0:
            evaluation_failures.append("private aggregate is missing")
        if int(angle_metrics.get("selected_task_count", 0)) <= 0:
            evaluation_failures.append("exact-angle evaluation is empty")
        if not {1.0, 3.0, 5.0, 10.0} <= set(angle_metrics.get("tolerances_degrees", [])):
            evaluation_failures.append("required angle tolerances are missing")
        if int(angle_metrics.get("private_row_outputs_written", -1)) != 0:
            evaluation_failures.append("private angle rows were written")
        angle_rows = read_csv_rows(report_root / "angle_estimation" / "angle_predictions.csv")
        if any(row.get("project_split") == PRIVATE_SPLIT or row.get("dataset") == "gmail" for row in angle_rows):
            evaluation_failures.append("private row present in public angle predictions")
        for row in angle_rows:
            if row.get("status") == "failure" and row.get("estimated_angle") not in {"", None}:
                evaluation_failures.append("failed angle silently received a numeric estimate")
                break
        checks.append({
            "name": "evaluation-outputs-valid",
            "passed": not evaluation_failures,
            "detail": "mapped, clustering, boundary, private aggregate, and exact-angle outputs validated" if not evaluation_failures else "; ".join(evaluation_failures),
        })
    except Exception as exc:
        checks.append({
            "name": "complete-pipeline-validation",
            "passed": False,
            "detail": f"{type(exc).__name__}: {exc}",
        })
    return checks


def _write_verification(cfg, checks, profile):
    passed = all(check["passed"] for check in checks)
    payload = {
        "all_passed": passed,
        "profile": profile,
        "checks": checks,
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    report_root = cfgmod.resolve_path(cfg, "reports") / "verification"
    atomic_write_json(report_root / "rotation_verification.json", payload)
    lines = [
        "# Rotation Pipeline Verification",
        "",
        f"Overall: {'PASS' if passed else 'FAIL'}",
        f"Profile: {profile}",
        "",
    ]
    for check in checks:
        lines.append(f"- [{'PASS' if check['passed'] else 'FAIL'}] {check['name']}: {check['detail']}")
    atomic_write_text(report_root / "verification_report.md", "\n".join(lines) + "\n")
    return payload


def _load_json(path):
    import json
    return json.loads(path.read_text(encoding="utf-8"))
