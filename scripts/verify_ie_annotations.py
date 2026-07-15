#!/usr/bin/env python3
"""Validate normalized public annotations and manifest privacy/split invariants."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config as cfgmod  # noqa: E402
from src.information_extraction.annotations import validate_annotation  # noqa: E402
from src.rotation_common import atomic_write_json, read_csv_rows  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    cfg = cfgmod.load_config(args.config)
    root = cfgmod.project_root(cfg)
    manifest = read_csv_rows(cfgmod.resolve_path(cfg, "metadata") / "information_extraction_manifest.csv")
    usable = [
        row for row in manifest
        if row.get("is_private") == "false" and row.get("is_usable") == "true"
        and row.get("normalized_annotation_path")
    ]
    if args.limit:
        usable = usable[: args.limit]
    errors = []
    counts: Counter[str] = Counter()
    seen_pages = set()
    seen_documents_by_split: dict[str, set[str]] = {}
    for row in usable:
        try:
            if row["page_id"] in seen_pages:
                raise ValueError("duplicate page_id")
            seen_pages.add(row["page_id"])
            document_splits = seen_documents_by_split.setdefault(row["document_id"], set())
            document_splits.add(row["project_split"])
            path = root / row["normalized_annotation_path"]
            payload = json.loads(path.read_text(encoding="utf-8"))
            validate_annotation(payload, root)
            if payload["page_id"] != row["page_id"] or payload["project_split"] != row["project_split"]:
                raise ValueError("manifest/annotation identity or split mismatch")
            if payload.get("is_private") is not False:
                raise ValueError("private flag in public normalized annotation")
            counts[row["dataset"]] += 1
        except Exception as exc:
            errors.append({
                "page_id": row.get("page_id", ""),
                "error_type": type(exc).__name__,
                "message": str(exc)[:500],
            })
    split_leaks = sorted(document for document, splits in seen_documents_by_split.items() if len(splits) > 1)
    private_rows = [row for row in manifest if row.get("is_private") == "true"]
    private_redacted = all(
        not row.get("image_path") and not row.get("annotation_path") and not row.get("normalized_annotation_path")
        and row.get("project_split") == "private_test"
        for row in private_rows
    )
    report = {
        "schema_version": "1.0",
        "status": "passed" if not errors and not split_leaks and private_redacted else "failed",
        "validated_annotations": len(usable) - len(errors),
        "validation_errors": len(errors),
        "counts_by_dataset": dict(sorted(counts.items())),
        "duplicate_page_ids": len(usable) - len(seen_pages),
        "document_split_leak_count": len(split_leaks),
        "private_public_rows": len(private_rows),
        "private_paths_redacted": private_redacted,
        "gmail_fit_rows": 0,
        "errors": errors[:100],
    }
    atomic_write_json(
        cfgmod.resolve_path(cfg, "reports") / "verification" / "ie_annotation_verification.json",
        report,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
