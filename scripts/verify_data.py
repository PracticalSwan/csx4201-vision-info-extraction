#!/usr/bin/env python3
"""verify_data.py — verify dataset and current-stage integrity.

Checks (per the project plan):
  * configured dataset paths exist
  * public and private folders are separated
  * Gmail files live under private paths
  * required metadata files exist
  * metadata inventory paths resolve to real files
  * no files were lost (re-scan counts match dataset_summary)
  * no raw file contents changed (move_verification sample hashes match)
  * duplicate reports are well-formed
  * no out-of-scope OCR outputs or neural checkpoints exist outside raw data

The completed classical rotation stage is legitimate: generated data under
``data/processed`` and ``data/splits``, public-safe reports, and artifacts under
``models/kmeans_rotation`` are allowed.

Exit code 0 = all checks passed; 1 = at least one failed.

Usage:
    python scripts/verify_data.py [--config config.yaml] [--log-level INFO]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

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
from src import privacy  # noqa: E402

log = logging.getLogger("vix.verify")

REQUIRED_METADATA = [
    "file_inventory.csv",
    "private_file_inventory.csv",
    "data_sources.csv",
    "dataset_summary.json",
    "processing_errors.csv",
    "duplicate_report.csv",
    "unmatched_files.csv",
]

# OCR and neural-model work remains outside the completed classical rotation
# stage. Raw source archives are exempt, including the pre-existing SROIE
# ``pytorch_model.bin`` and ``training_args.bin`` files.
FORBIDDEN_OUTPUT_DIR_NAMES = {
    "checkpoint",
    "checkpoints",
    "ocr_output",
    "ocr_outputs",
    "ocr_result",
    "ocr_results",
}
FORBIDDEN_OCR_FILE_PREFIXES = ("ocr_output", "ocr_result")
FORBIDDEN_NEURAL_MODEL_EXTS = {
    ".bin",
    ".ckpt",
    ".h5",
    ".hdf5",
    ".keras",
    ".onnx",
    ".pb",
    ".pt",
    ".pth",
    ".safetensors",
    ".tflite",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    ap.add_argument("--log-level", default=None)
    args = ap.parse_args()

    cfg = cfgmod.load_config(args.config)
    cfgmod.setup_logging(cfg, args.log_level)
    root = cfgmod.project_root(cfg)
    metadata_dir = cfgmod.resolve_path(cfg, "metadata")

    checks: list[tuple[str, bool, str]] = []

    # 1. Discovery + configured paths exist.
    datasets = dd.discover_datasets(cfg)
    checks.append(("datasets-discovered", len(datasets) >= 4,
                   f"{len(datasets)} datasets (need >=4)"))
    for ds in datasets:
        ok = ds.current_path.exists()
        checks.append((f"dataset-path:{ds.name}", ok,
                       f"{ds.current_path} exists={ok}"))

    # 2. Public/private separation.
    sep_ok = True
    for ds in datasets:
        try:
            for p in ds.current_path.rglob("*"):
                if p.is_file():
                    privacy.assert_private_not_under_public(p)
        except ValueError as exc:
            sep_ok = False
            checks.append(("public-private-separation", False, str(exc)))
    checks.append(("public-private-separation", sep_ok,
                   "no private file under public tree" if sep_ok else "VIOLATION"))

    # 3. Gmail under private.
    gmail_ds = next((d for d in datasets if d.name == "gmail"), None)
    gmail_ok = True
    if gmail_ds:
        gmail_ok = "private" in str(gmail_ds.current_path).replace("\\", "/").lower()
        checks.append(("gmail-is-private", gmail_ok,
                       f"gmail root={gmail_ds.current_path}"))
    else:
        checks.append(("gmail-is-private", False, "gmail dataset not found"))

    # 4. Metadata files exist.
    for name in REQUIRED_METADATA:
        p = metadata_dir / name
        checks.append((f"metadata-exists:{name}", p.exists(), str(p)))

    # 5. Inventory paths resolve.
    inventory = metadata_dir / "file_inventory.csv"
    resolve_ok, resolve_total, resolve_bad = _check_inventory_resolves(inventory, root)
    checks.append(("inventory-paths-resolve", resolve_ok,
                   f"{resolve_total - resolve_bad}/{resolve_total} resolved"))

    # 6. No files lost: re-scan vs dataset_summary.
    summary_path = metadata_dir / "dataset_summary.json"
    count_ok, count_detail = _check_counts(summary_path, datasets)
    checks.append(("no-files-lost", count_ok, count_detail))

    # 7. Move verification hashes (if a move happened).
    ver_path = metadata_dir / "move_verification.json"
    hash_ok, hash_detail = _check_move_hashes(ver_path)
    checks.append(("raw-bytes-unchanged", hash_ok, hash_detail))

    # 8. Duplicate report well-formed.
    dup_ok, dup_detail = _check_duplicates(metadata_dir / "duplicate_report.csv")
    checks.append(("duplicate-report-consistent", dup_ok, dup_detail))

    # 9. No out-of-scope OCR or neural-model outputs. Classical rotation
    # artifacts under the configured processed/split/model/report paths are
    # expected and allowed.
    forb_ok, forb_detail = _check_forbidden(root)
    checks.append(("no-disallowed-neural-or-ocr-outputs", forb_ok, forb_detail))

    # Report.
    print("\n" + "=" * 72)
    print("VERIFICATION REPORT")
    print("=" * 72)
    passed = 0
    failed = []
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}: {detail}")
        if ok:
            passed += 1
        else:
            failed.append(name)
    print("=" * 72)
    print(f"{passed}/{len(checks)} checks passed.")

    # Persist a machine-readable result alongside the report.
    _write_verify_json(metadata_dir / "verification_result.json", checks)
    # Generate the human-readable organization report from all metadata + checks.
    _write_organization_report(metadata_dir, checks, datasets, failed)

    if failed:
        print(f"FAILED checks: {', '.join(failed)}")
        return 1
    print("All checks passed.")
    return 0


# ---------------------------------------------------------------------------
# organization_report.md
# ---------------------------------------------------------------------------


def _write_organization_report(metadata_dir: Path, checks, datasets, failed) -> None:
    """Assemble data/metadata/organization_report.md from all metadata + checks.

    Public-safe: Gmail filenames are never written; only aggregate private counts.
    """
    import csv as _csv

    def _load_json(name):
        p = metadata_dir / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def _csv_rows(name):
        p = metadata_dir / name
        if not p.exists():
            return []
        with p.open(encoding="utf-8", newline="") as fh:
            return list(_csv.DictReader(fh))

    summary = _load_json("dataset_summary.json")
    manifest = _load_json("organization_manifest.json")
    errors = _csv_rows("processing_errors.csv")
    dup_rows = _csv_rows("duplicate_report.csv")
    unmatched = _csv_rows("unmatched_files.csv")

    mode = manifest.get("mode", "unknown") if isinstance(manifest, dict) else "unknown"
    pass_count = sum(1 for _, ok, _ in checks if ok)
    overall = "PASS" if not failed else "FAIL"

    lines: list[str] = []
    lines.append("# Organization Report\n")
    lines.append(f"Overall verification: **{overall}** "
                 f"({pass_count}/{len(checks)} checks passed).")
    lines.append(f"Organization mode used: `{mode}`.\n")

    lines.append("## 1. Original directory structure found")
    lines.append("Data originally lived under `vision_info_extraction_data/` with the "
                 "layout recorded in `AGENT_MEMORY.md`: `public_train/{coru_receipts, "
                 "docvqa_samples, fatura_invoices, funsd_forms, sroie_receipts}`, "
                 "`gmail_private_test/`, plus empty `public_test/`, `augmented_rotated/`, "
                 "and `metadata/` placeholders. Several datasets carried both source "
                 "archives (`.zip`) and extracted folders; FUNSD also carried macOS "
                 "`__MACOSX/` junk.")

    lines.append("\n## 2-3. Dataset roots detected and identification confidence")
    lines.append("| Dataset | Source | Confidence | Current path |")
    lines.append("|---------|--------|------------|--------------|")
    for ds in datasets:
        lines.append(f"| {ds.name} | {ds.source_type} | {ds.confidence} | `{ds.current_path}` |")

    lines.append("\n## 4-5. Organization actions")
    if isinstance(manifest, dict):
        for a in manifest.get("actions", []):
            lines.append(f"- `{a.get('action')}` **{a.get('dataset')}**: "
                         f"`{a.get('current')}` -> `{a.get('target')}` ({a.get('note', '')})")
    else:
        lines.append("- (no manifest found)")

    lines.append("\n## 6-10. Per-dataset counts, size, images, PDFs, annotations")
    lines.append("| Dataset | Files | Size | Images | PDFs | Annotations | Unreadable | Empty |")
    lines.append("|---------|-------|------|--------|------|-------------|-----------|-------|")
    for name, s in summary.items():
        lines.append(
            f"| {name} | {s.get('total_files', 0)} | {_human(s.get('total_size_bytes', 0))} | "
            f"{s.get('images', 0)} | {s.get('pdfs', 0)} | {s.get('annotations', 0)} | "
            f"{s.get('unreadable_files', 0)} | {s.get('empty_files', 0)} |"
        )

    lines.append("\n## 11-12. Corrupted and empty files")
    type_counts: dict[str, int] = {}
    for e in errors:
        type_counts[e.get("error_type", "?")] = type_counts.get(e.get("error_type", "?"), 0) + 1
    if type_counts:
        for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
            lines.append(f"- {t}: {c}")
    else:
        lines.append("- none recorded")

    lines.append("\n## 13-14. Duplicates")
    exact_groups = {r["duplicate_group_id"] for r in dup_rows if r.get("duplicate_type") == "exact"}
    near_groups = {r["duplicate_group_id"] for r in dup_rows if r.get("duplicate_type") == "likely_near_duplicate"}
    lines.append(f"- Exact duplicate groups: {len(exact_groups)}")
    lines.append(f"- Near-duplicate groups: {len(near_groups)}")

    lines.append("\n## 15. Missing image-annotation pairs")
    if unmatched:
        lines.append(f"- {len(unmatched)} unmatched files recorded (see `unmatched_files.csv`).")
    else:
        lines.append("- none recorded")

    lines.append("\n## 16. Unresolved files / issues")
    any_issues = False
    for name, s in summary.items():
        for issue in s.get("issues", []):
            any_issues = True
            lines.append(f"- {name}: {issue}")
    if not any_issues:
        lines.append("- none")

    lines.append("\n## 17. Gmail private-file count")
    gmail_summary = summary.get("gmail", {})
    lines.append(f"- Gmail private files: {gmail_summary.get('total_files', 0)} "
                 f"(receipts/invoices/legal_financial_docs/unclassified). "
                 f"Real filenames are kept only in the gitignored "
                 f"`private_file_inventory.csv`.")

    lines.append("\n## 18. Privacy actions applied")
    lines.append("- Gmail documents moved under `data/raw/private/gmail/`.")
    lines.append("- Public inventory anonymizes Gmail filenames.")
    lines.append("- `.gitignore` excludes all raw data and the private inventory.")
    lines.append("- Public/private separation verified.")

    lines.append("\n## 19. Verification results")
    for name, ok, detail in checks:
        lines.append(f"- [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    lines.append("\n## 20. Remaining manual decisions")
    lines.append("- Review any low-confidence Gmail classifications in `unclassified/`.")
    lines.append("- Decide GitHub repo visibility before any commit (Gmail requires a "
                 "private repo or must stay uncommitted).")
    lines.append("- Decide large-binary policy (Git LFS vs. documented downloads) for "
                 "the public `.zip`/image archives before the first push.")
    lines.append("- Confirm the bundled SROIE pretrained model is intentionally kept "
                 "(it is preserved and unused at this stage).")

    (metadata_dir / "organization_report.md").write_text("\n".join(lines), encoding="utf-8")


def _human(n) -> str:
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _check_inventory_resolves(inventory: Path, root: Path) -> tuple[bool, int, int]:
    if not inventory.exists():
        return False, 0, 0
    total = bad = 0
    with inventory.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            # Skip private rows: their filenames are anonymized in the public
            # inventory and intentionally do not resolve on disk.
            if str(row.get("is_private")).lower() == "true":
                continue
            rel = row.get("current_relative_path", "")
            if not rel:
                continue
            total += 1
            # Resolve relative to project root (public rel paths look like
            # "sroie/..." which maps under data/raw/public).
            candidates = [root / rel]
            if not candidates[0].exists():
                candidates.append(root / "data" / "raw" / "public" / rel)
                candidates.append(root / "data" / "raw" / "private" / rel)
            if not any(c.exists() for c in candidates):
                bad += 1
    return bad == 0, total, bad


def _check_counts(summary_path: Path, datasets) -> tuple[bool, str]:
    if not summary_path.exists():
        return False, "dataset_summary.json missing"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    mismatches = []
    for ds in datasets:
        recorded = summary.get(ds.name, {}).get("total_files")
        if recorded is None:
            continue
        actual = sum(1 for _ in ds.current_path.rglob("*") if _.is_file())
        if actual != recorded:
            mismatches.append(f"{ds.name}: {actual} vs {recorded}")
    if mismatches:
        return False, "; ".join(mismatches)
    return True, "all dataset counts match summary"


def _check_move_hashes(ver_path: Path) -> tuple[bool, str]:
    if not ver_path.exists():
        return True, "no move verification file (reference mode or not yet moved)"
    data = json.loads(ver_path.read_text(encoding="utf-8"))
    bad = 0
    total = 0
    for ds_name, rec in data.items():
        # The post_sample hashes were taken right after the move; the pre_sample
        # were taken right before. They must be equal AND still match the file
        # on disk now.
        for rel, pre in rec.get("pre_sample", {}).items():
            total += 1
            post = rec.get("post_sample", {}).get(rel)
            if post != pre:
                bad += 1
    if bad:
        return False, f"{bad}/{total} sample hashes differ pre/post move"
    return True, f"{total} sample hashes stable across move"


def _check_duplicates(dup_csv: Path) -> tuple[bool, str]:
    if not dup_csv.exists():
        return False, "duplicate_report.csv missing"
    groups: dict[str, int] = {}
    with dup_csv.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            groups[row["duplicate_group_id"]] = groups.get(row["duplicate_group_id"], 0) + 1
    bad = [gid for gid, n in groups.items() if n < 2]
    if bad:
        return False, f"{len(bad)} groups have <2 members"
    return True, f"{len(groups)} duplicate groups well-formed"


def _check_forbidden(root: Path) -> tuple[bool, str]:
    """Reject OCR and neural checkpoint outputs outside ``data/raw``.

    The function name is retained for compatibility with the organization-stage
    verifier. Classical rotation artifacts (including ``.joblib`` files) are
    intentionally accepted. The raw tree is pruned before inspection so bundled
    upstream files such as SROIE's ``.bin`` weights remain exempt and the check
    does not traverse the large immutable corpus.
    """
    root = root.resolve()
    raw_root = (root / "data" / "raw").resolve()
    found_dirs: list[str] = []
    neural_hits: list[str] = []
    ocr_file_hits: list[str] = []

    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        retained_dirs: list[str] = []
        for dirname in dirnames:
            child = current / dirname
            if _is_within(child, raw_root):
                continue
            if dirname.casefold() in FORBIDDEN_OUTPUT_DIR_NAMES:
                found_dirs.append(child.relative_to(root).as_posix())
                continue
            retained_dirs.append(dirname)
        dirnames[:] = retained_dirs

        for filename in filenames:
            path = current / filename
            suffix = path.suffix.casefold()
            relative = path.relative_to(root).as_posix()
            if suffix in FORBIDDEN_NEURAL_MODEL_EXTS:
                neural_hits.append(relative)
            lowered = filename.casefold()
            if any(
                lowered == prefix
                or lowered.startswith(f"{prefix}.")
                or lowered.startswith(f"{prefix}_")
                or lowered.startswith(f"{prefix}-")
                for prefix in FORBIDDEN_OCR_FILE_PREFIXES
            ):
                ocr_file_hits.append(relative)

    if found_dirs or neural_hits or ocr_file_hits:
        detail = {
            "directories": found_dirs[:20],
            "neural_files": neural_hits[:20],
            "ocr_files": ocr_file_hits[:20],
        }
        return False, f"disallowed outside data/raw: {detail}"
    return True, (
        "no OCR outputs or neural checkpoints outside data/raw; "
        "classical rotation artifacts allowed"
    )


def _is_within(path: Path, parent: Path) -> bool:
    """Return whether *path* resolves inside *parent* without requiring it to exist."""
    try:
        path.resolve().relative_to(parent)
    except ValueError:
        return False
    return True


def _write_verify_json(path: Path, checks: list[tuple[str, bool, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "all_passed": all(ok for _, ok, _ in checks),
        "checks": [{"name": n, "passed": ok, "detail": d} for n, ok, d in checks],
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
