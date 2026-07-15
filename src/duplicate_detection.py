"""Exact and near-duplicate detection.

Exact duplicates use SHA-256 (identical bytes). Near-duplicate images use
perceptual hashing (imagehash.phash) with a configurable Hamming threshold and
a cap on how many images receive the full pairwise scan, so the audit stays
tractable on large corpora. Duplicates are never deleted; only a recommended
canonical file is suggested.
"""
from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

log = logging.getLogger("vix.dupes")

DUPLICATE_COLUMNS = [
    "duplicate_group_id",
    "dataset",
    "file_path",
    "sha256",
    "size_bytes",
    "duplicate_type",
    "recommended_canonical_file",
    "cross_dataset_duplicate",
    "notes",
]


def find_exact_duplicates(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group rows that share a non-empty SHA-256 (byte-identical files)."""
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        sha = r.get("sha256")
        if sha:
            by_hash[sha].append(r)
    return [group for group in by_hash.values() if len(group) > 1]


def find_near_duplicates(
    rows: list[dict[str, Any]],
    cfg: Mapping[str, Any],
    *,
    base_dir: Path | None = None,
) -> list[list[dict[str, Any]]]:
    """Group visually similar images via perceptual hash.

    To keep the pairwise comparison tractable we bucket images by their
    (width, height); near-duplicates from the same source are typically the
    same dimensions, and this bounds the comparisons. The total number of
    images hashed is capped at ``max_images_for_full_near_duplicate_scan``;
    beyond that a warning is logged and the scan is truncated (never silently).
    """
    if not cfg.get("duplicates", {}).get("perceptual_enabled", True):
        return []
    try:
        import imagehash
        from PIL import Image
    except ImportError:  # pragma: no cover
        log.warning("imagehash/Pillow unavailable; skipping near-duplicate scan")
        return []

    threshold = int(cfg.get("duplicates", {}).get("perceptual_threshold", 5))
    cap = int(cfg.get("duplicates", {}).get("max_images_for_full_near_duplicate_scan", 10000))
    # Bound the per-bucket pairwise work. Buckets larger than this are truncated
    # (sorted by path) and the truncation is logged so the scan always finishes.
    max_bucket = 1500

    image_rows = [r for r in rows if r.get("is_image") and r.get("is_readable")]
    if len(image_rows) > cap:
        log.warning(
            "near-duplicate scan capped: %d images exceed limit %d; scanning first %d only",
            len(image_rows), cap, cap,
        )
        image_rows = image_rows[:cap]

    # Compute phash per image, bucketed by dimensions.
    buckets: dict[tuple[int, int], list[tuple[Any, dict[str, Any]]]] = defaultdict(list)
    for r in image_rows:
        path = _resolve(r, base_dir)
        if path is None or not path.exists():
            continue
        try:
            with Image.open(path) as im:
                dims = (int(im.width), int(im.height))
                phash = imagehash.phash(im)
        except Exception:
            continue
        buckets[dims].append((phash, r))

    groups: list[list[dict[str, Any]]] = []
    for dims, bucket in buckets.items():
        if len(bucket) > max_bucket:
            log.warning(
                "near-dup bucket (%dx%d) has %d images > limit %d; "
                "scanning first %d only (sorted by path)",
                dims[0], dims[1], len(bucket), max_bucket, max_bucket,
            )
            bucket = sorted(bucket, key=lambda x: x[1]["current_relative_path"])[:max_bucket]
        groups.extend(_cluster_by_hamming(bucket, threshold))
    # Merge groups that share any member (transitive near-duplication).
    return _merge_overlapping(groups)


def _cluster_by_hamming(
    bucket: list[tuple[Any, dict[str, Any]]], threshold: int
) -> list[list[dict[str, Any]]]:
    """Greedy union of images whose phash Hamming distance <= threshold."""
    parent = list(range(len(bucket)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(len(bucket)):
        for j in range(i + 1, len(bucket)):
            if (bucket[i][0] - bucket[j][0]) <= threshold:
                union(i, j)

    clusters: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for idx, (_, row) in enumerate(bucket):
        clusters[find(idx)].append(row)
    return [c for c in clusters.values() if len(c) > 1]


def _merge_overlapping(
    groups: list[list[dict[str, Any]]],
) -> list[list[dict[str, Any]]]:
    merged: list[set[str]] = []
    key_of: dict[str, int] = {}
    for group in groups:
        ids = {r["file_id"] for r in group}
        overlap = {key_of[i] for i in ids if i in key_of}
        if not overlap:
            idx = len(merged)
            merged.append(set())
            target = idx
        else:
            target = next(iter(overlap))
        for i in ids:
            key_of[i] = target
            merged[target].update(ids)
    result = []
    seen: set[str] = set()
    for ids in merged:
        if not ids - seen:
            continue
        result.append(ids)
        seen.update(ids)
    # Map back to rows (deduped).
    row_by_id: dict[str, dict[str, Any]] = {}
    for group in groups:
        for r in group:
            row_by_id.setdefault(r["file_id"], r)
    return [[row_by_id[i] for i in sorted(g)] for g in result if len(g) > 1]


def _resolve(row: dict[str, Any], base_dir: Path | None) -> Path | None:
    abs_path = row.get("_abs_path")
    if abs_path:
        return Path(abs_path)
    rel = row.get("current_relative_path")
    if not rel:
        return None
    if base_dir is not None:
        return base_dir / rel
    return Path(rel)


def build_duplicate_report(
    exact: list[list[dict[str, Any]]],
    near: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Flatten duplicate groups into report rows.

    The recommended canonical file is the first member of each group when sorted
    by (dataset, current_relative_path) — a non-destructive suggestion only.
    """
    out: list[dict[str, Any]] = []
    for kind, groups in (("exact", exact), ("likely_near_duplicate", near)):
        for gid, group in enumerate(groups, start=1):
            group_sorted = sorted(group, key=lambda r: (r["dataset"], r["current_relative_path"]))
            canonical = group_sorted[0]["current_relative_path"]
            datasets = {r["dataset"] for r in group}
            cross = "yes" if len(datasets) > 1 else "no"
            for r in group_sorted:
                out.append({
                    "duplicate_group_id": f"{kind[0]}{gid:05d}",
                    "dataset": r["dataset"],
                    "file_path": r["current_relative_path"],
                    "sha256": r.get("sha256", ""),
                    "size_bytes": r.get("size_bytes", 0),
                    "duplicate_type": kind,
                    "recommended_canonical_file": canonical,
                    "cross_dataset_duplicate": cross,
                    "notes": "" if r["current_relative_path"] == canonical else "candidate for de-dup",
                })
    return out


def write_duplicate_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=DUPLICATE_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def count_exact_groups(exact: Iterable[list[dict[str, Any]]]) -> int:
    return sum(1 for _ in exact)
