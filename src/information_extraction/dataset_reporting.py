"""Aggregate-only reporting for the immutable final public model dataset."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.rotation_common import atomic_write_csv, atomic_write_json, atomic_write_text, read_csv_rows, sha256_file


EXCLUSION_COLUMNS = (
    "page_id",
    "dataset",
    "project_split",
    "token_source",
    "exclusion_reason",
)


def generate_final_dataset_reports(
    manifest_path: str | Path,
    split_manifest_path: str | Path,
    report_root: str | Path,
) -> dict[str, Any]:
    """Write the required privacy-safe final-corpus reports from executed artifacts."""
    manifest = read_csv_rows(Path(manifest_path))
    split_rows = read_csv_rows(Path(split_manifest_path))
    if not manifest:
        raise ValueError("final model manifest is empty")
    if {row.get("profile", "") for row in manifest} != {"final"}:
        raise ValueError("dataset reports require an exclusively final-profile manifest")
    if any(row.get("is_private", "").casefold() != "false" for row in manifest):
        raise ValueError("private or unmarked rows are refused from public dataset reports")

    report_dir = Path(report_root)
    report_dir.mkdir(parents=True, exist_ok=True)
    usable = [row for row in manifest if row.get("is_usable") == "true"]
    exclusions = [
        {column: row.get(column, "") for column in EXCLUSION_COLUMNS}
        for row in manifest
        if row.get("is_usable") != "true"
    ]
    atomic_write_csv(report_dir / "exclusions.csv", exclusions, EXCLUSION_COLUMNS)

    normalized_pages: Counter[str] = Counter()
    split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in split_rows:
        if row.get("is_private", "").casefold() != "false":
            continue
        dataset = row.get("dataset", "unknown")
        normalized_pages[dataset] += 1
        split_counts[dataset][row.get("project_split", "unknown")] += 1

    usable_page_sets: dict[str, set[str]] = defaultdict(set)
    examples_by_dataset: Counter[str] = Counter()
    streams_by_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    for row in usable:
        dataset = row.get("dataset", "unknown")
        examples_by_dataset[dataset] += 1
        streams_by_dataset[dataset][row.get("token_source", "unknown")] += 1
        if row.get("token_source") == "ground_truth":
            usable_page_sets[dataset].add(row.get("page_id", ""))

    alignment = _alignment_metrics(usable)
    dataset_names = sorted(set(normalized_pages) | set(examples_by_dataset))
    by_dataset: dict[str, Any] = {}
    for dataset in dataset_names:
        normalized = normalized_pages[dataset]
        usable_pages = len(usable_page_sets[dataset])
        by_dataset[dataset] = {
            "normalized_public_pages": normalized,
            "usable_fit_pages": usable_pages,
            "excluded_fit_pages": max(0, normalized - usable_pages),
            "usable_examples": examples_by_dataset[dataset],
            "streams": dict(sorted(streams_by_dataset[dataset].items())),
            "split_counts": dict(sorted(split_counts[dataset].items())),
            "alignment": alignment["by_dataset"].get(dataset),
        }

    build_ids = {row.get("build_id", "") for row in manifest}
    if len(build_ids) != 1 or "" in build_ids:
        raise ValueError("final model manifest has missing or mixed build IDs")
    summary = {
        "schema_version": "1.0",
        "profile": "final",
        "build_id": next(iter(build_ids)),
        "manifest_path": str(Path(manifest_path).resolve()),
        "manifest_sha256": sha256_file(Path(manifest_path)),
        "split_manifest_path": str(Path(split_manifest_path).resolve()),
        "split_manifest_sha256": sha256_file(Path(split_manifest_path)),
        "normalized_public_pages": sum(normalized_pages.values()),
        "usable_public_fit_pages": sum(len(values) for values in usable_page_sets.values()),
        "usable_examples": len(usable),
        "excluded_examples": len(exclusions),
        "gmail_fit_rows": 0,
        "private_fit_rows": 0,
        "datasets": by_dataset,
    }
    atomic_write_json(report_dir / "dataset_summary.json", summary)
    atomic_write_json(report_dir / "alignment_metrics.json", alignment)
    atomic_write_text(report_dir / "dataset_quality.md", _quality_markdown(summary, alignment))
    return summary


def _alignment_metrics(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    accumulators: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    canonical_fields: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        stream = row.get("token_source", "")
        if stream not in {"paddleocr", "hybrid"}:
            continue
        path = Path(row.get("model_example_path", ""))
        if not path.is_file():
            raise FileNotFoundError(path)
        example = json.loads(path.read_text(encoding="utf-8"))
        if example.get("is_private") is not False:
            raise ValueError(f"private or unmarked example refused from reporting: {path}")
        key = (row.get("dataset", "unknown"), stream)
        bucket = accumulators[key]
        source_tokens = len(example.get("source_targets", {}).get("tokens", []))
        unmatched_labels = len(example.get("alignment", {}).get("unmatched_labels", []))
        matched_tokens = max(0, source_tokens - unmatched_labels)
        bucket["pages"] += 1
        bucket["source_tokens"] += source_tokens
        bucket["matched_tokens"] += matched_tokens
        bucket["unmatched_tokens"] += unmatched_labels
        bucket["unmatched_ocr_tokens"] += len(
            example.get("alignment", {}).get("unmatched_ocr_tokens", [])
        )
        for metric in (
            "alignment_coverage",
            "entity_retention_rate",
            "relation_retention_rate",
            "canonical_retention_rate",
            "data_quality_score",
        ):
            bucket[metric] += float(example.get(metric, 0.0))
        for name, field in example.get("canonical_fields", {}).items():
            if not isinstance(field, Mapping) or field.get("evidence_valid", True):
                canonical_fields[key][str(name)] += 1

    by_dataset: dict[str, dict[str, Any]] = defaultdict(dict)
    for (dataset, stream), bucket in sorted(accumulators.items()):
        pages = int(bucket["pages"])
        by_dataset[dataset][stream] = {
            "pages": pages,
            "source_tokens": int(bucket["source_tokens"]),
            "matched_tokens": int(bucket["matched_tokens"]),
            "unmatched_tokens": int(bucket["unmatched_tokens"]),
            "unmatched_ocr_tokens": int(bucket["unmatched_ocr_tokens"]),
            **{
                f"mean_{metric}": bucket[metric] / pages if pages else 0.0
                for metric in (
                    "alignment_coverage",
                    "entity_retention_rate",
                    "relation_retention_rate",
                    "canonical_retention_rate",
                    "data_quality_score",
                )
            },
            "field_availability": dict(sorted(canonical_fields[(dataset, stream)].items())),
        }
    return {
        "schema_version": "1.0",
        "scope": "public OCR-variant examples only; ground-truth stream is immutable",
        "by_dataset": dict(by_dataset),
    }


def _quality_markdown(summary: Mapping[str, Any], alignment: Mapping[str, Any]) -> str:
    lines = [
        "# Final model dataset quality",
        "",
        f"Build: `{summary['build_id']}`.",
        f"Normalized public pages: {summary['normalized_public_pages']}.",
        f"Usable public fit pages: {summary['usable_public_fit_pages']}.",
        f"Usable examples across streams: {summary['usable_examples']}.",
        "Private/Gmail fit rows: **0**.",
        "",
        "The ground-truth stream is immutable. PaddleOCR examples are inference-realistic; hybrid examples are training-only and preserve supervised targets with explicit masks.",
        "CORU remains wholly unseen-domain evaluation data because it has QA answers but no compatible token polygons.",
        "",
        "| Dataset | Normalized | Fit pages | Examples | Splits |",
        "|---|---:|---:|---:|---|",
    ]
    for dataset, values in sorted(summary["datasets"].items()):
        splits = ", ".join(f"{key}={value}" for key, value in values["split_counts"].items())
        lines.append(
            f"| {dataset} | {values['normalized_public_pages']} | "
            f"{values['usable_fit_pages']} | {values['usable_examples']} | {splits} |"
        )
    lines.extend(["", "Alignment and target-retention details are machine-readable in `alignment_metrics.json`."])
    if not alignment["by_dataset"]:
        lines.append("No OCR-variant examples were present.")
    return "\n".join(lines) + "\n"
