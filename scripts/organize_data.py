#!/usr/bin/env python3
"""organize_data.py — organize or reference datasets into the target structure.

Modes:
    reference (default) — leave raw data in place; write a manifest mapping each
                          canonical target to its current location. Nothing is moved.
    move                — move each dataset's current folder into its canonical
                          target (safe on the same volume: a metadata rename).
                          File counts and a sample of SHA-256 hashes are verified
                          before and after the move.
    copy                — copy each dataset into its canonical target (duplicates
                          data; use only when source and target differ in volume).

Options:
    --config config.yaml   configuration file
    --dry-run              plan only; do not move/copy/write manifests
    --force                allow overwriting an empty existing target
    --log-level LEVEL      logging level

The script never deletes raw files and never modifies file contents.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
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

log = logging.getLogger("vix.organize")

SAMPLE_HASH_COUNT = 20


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("reference", "move", "copy"),
                    default=None, help="override organization.default_mode")
    ap.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--log-level", default=None)
    args = ap.parse_args()

    cfg = cfgmod.load_config(args.config)
    cfgmod.setup_logging(cfg, args.log_level)
    mode = args.mode or cfg["organization"]["default_mode"]
    verify = bool(cfg["organization"].get("verify_after_move", True))

    datasets = dd.discover_datasets(cfg)
    if not datasets:
        log.error("No datasets discovered; nothing to organize.")
        return 1

    metadata_dir = cfgmod.resolve_path(cfg, "metadata")
    manifest_path = metadata_dir / "organization_manifest.json"
    log.info("Mode=%s dry_run=%s datasets=%d", mode, args.dry_run, len(datasets))

    actions: list[dict] = []
    for ds in datasets:
        actions.extend(organize_one(ds, mode, args.dry_run, args.force, verify))

    if not args.dry_run:
        metadata_dir.mkdir(parents=True, exist_ok=True)
        # Ensure the canonical directory tree exists (placeholders for reference
        # mode; actual data lands here for move/copy).
        _ensure_skeleton(cfg)
        # Pull structured verification records out before writing the manifest.
        verification = {a["dataset"]: a.pop("_verification")
                        for a in actions if "_verification" in a}
        with manifest_path.open("w", encoding="utf-8") as fh:
            json.dump({
                "mode": mode,
                "generated_at": datetime.now(tz=timezone.utc).isoformat(),
                "actions": actions,
            }, fh, indent=2)
        log.info("Wrote manifest: %s", manifest_path)
        if verification:
            public_verification, private_verification = _privacy_safe_verification(verification)
            ver_path = metadata_dir / "move_verification.json"
            with ver_path.open("w", encoding="utf-8") as fh:
                json.dump(public_verification, fh, indent=2)
            log.info("Wrote move verification: %s", ver_path)
            if private_verification:
                private_ver_path = metadata_dir / "private_move_verification.json"
                with private_ver_path.open("w", encoding="utf-8") as fh:
                    json.dump(private_verification, fh, indent=2)
                log.info("Wrote private move verification: %s", private_ver_path)

    print(f"\nOrganization mode: {mode}  (dry_run={args.dry_run})")
    for a in actions:
        print(f"  [{a['action']:<8}] {a['dataset']:<7} {a['current']} -> {a['target']}  {a.get('note', '')}")
    return 0


def organize_one(ds: dd.DatasetInfo, mode: str, dry_run: bool,
                 force: bool, verify: bool) -> list[dict]:
    src = ds.current_path
    dst = ds.target_path
    action = {"dataset": ds.name, "current": str(src), "target": str(dst)}

    if src.resolve() == dst.resolve() or (dst.exists() and src.exists()
                                          and _same_file_counts(src, dst)):
        action.update(action_ref(action_str("already-organized", mode), ds))
        return [action]

    if dry_run:
        action.update(action_ref(_planned_action(mode), ds))
        return [action]

    if mode == "reference":
        # Data stays put; we only record the mapping.
        action.update(action_ref("referenced", ds))
        return [action]

    if mode == "move":
        return _do_move(ds, src, dst, force, verify)
    if mode == "copy":
        return _do_copy(ds, src, dst, force, verify)
    raise ValueError(f"unknown mode {mode}")


def _do_move(ds, src, dst, force, verify) -> list[dict]:
    if not src.exists():
        log.warning("[%s] source missing: %s", ds.name, src)
        return [{"dataset": ds.name, "current": str(src), "target": str(dst),
                 "action": "skipped", "note": "source missing"}]
    _prepare_target(dst, force)
    pre_count = _count_files(src)
    sample = _sample_hashes(src, SAMPLE_HASH_COUNT) if verify else []
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    post_count = _count_files(dst)
    note = f"files {pre_count}->{post_count}"
    verification = {"pre_sample": dict(sample), "post_sample": {}, "count_ok": pre_count == post_count}
    if pre_count != post_count:
        log.error("[%s] MOVE COUNT MISMATCH %d != %d", ds.name, pre_count, post_count)
        note += " COUNT MISMATCH"
    if verify:
        mismatches = _verify_hashes(dst, sample)
        verification["post_sample"] = {rel: _sha256(dst / rel) for rel, _ in sample}
        verification["hash_mismatches"] = mismatches
        if mismatches:
            log.error("[%s] MOVE HASH MISMATCH on %d files", ds.name, len(mismatches))
            note += f" HASH MISMATCH x{len(mismatches)}"
        else:
            note += " hashes-ok"
    log.info("[%s] moved %s -> %s (%s)", ds.name, src, dst, note)
    return [{"dataset": ds.name, "current": str(src), "target": str(dst),
             "action": "moved", "note": note, "_verification": verification}]


def _do_copy(ds, src, dst, force, verify) -> list[dict]:
    if not src.exists():
        return [{"dataset": ds.name, "current": str(src), "target": str(dst),
                 "action": "skipped", "note": "source missing"}]
    _prepare_target(dst, force)
    pre_count = _count_files(src)
    sample = _sample_hashes(src, SAMPLE_HASH_COUNT) if verify else []
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(str(src), str(dst))
    post_count = _count_files(dst)
    note = f"files {pre_count}->{post_count}"
    if verify:
        mismatches = _verify_hashes(dst, sample)
        note += " hashes-ok" if not mismatches else f" HASH MISMATCH x{len(mismatches)}"
    log.info("[%s] copied %s -> %s (%s)", ds.name, src, dst, note)
    return [{"dataset": ds.name, "current": str(src), "target": str(dst),
             "action": "copied", "note": note}]


def _prepare_target(dst: Path, force: bool) -> None:
    if dst.exists():
        if force and dst.is_dir() and not any(dst.iterdir()):
            dst.rmdir()
            return
        if force:
            raise RuntimeError(f"--force refuses to clobber non-empty target: {dst}")
        raise FileExistsError(f"target already exists: {dst} (use --force only empties)")


def _count_files(root: Path) -> int:
    return sum(1 for _ in root.rglob("*") if _.is_file())


def _same_file_counts(a: Path, b: Path) -> bool:
    try:
        return _count_files(a) == _count_files(b)
    except OSError:
        return False


def _sample_hashes(root: Path, n: int) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    files = sorted((p for p in root.rglob("*") if p.is_file()),
                   key=lambda p: str(p))[:n]
    for p in files:
        out.append((p.relative_to(root).as_posix(), _sha256(p)))
    return out


def _verify_hashes(root: Path, sample: list[tuple[str, str]]) -> list[str]:
    bad: list[str] = []
    for rel, expected in sample:
        p = root / rel
        if not p.exists() or _sha256(p) != expected:
            bad.append(rel)
    return bad


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _privacy_safe_verification(verification: dict) -> tuple[dict, dict]:
    """Keep private filenames out of the committable verification artifact.

    Hash values remain unchanged.  Only the identity-bearing relative-path keys
    for the Gmail dataset are replaced by deterministic opaque sample IDs.  The
    original keyed record is retained in a separately ignored private file.
    """
    public = {}
    private = {}
    for dataset, record in verification.items():
        if dataset != "gmail":
            public[dataset] = record
            continue
        private[dataset] = record
        safe_record = dict(record)
        for field in ("pre_sample", "post_sample"):
            safe_record[field] = {
                f"private_sample_{hashlib.sha256(rel.encode('utf-8')).hexdigest()[:12]}": digest
                for rel, digest in record.get(field, {}).items()
            }
        public[dataset] = safe_record
    return public, private


def _ensure_skeleton(cfg) -> None:
    for key in ("sroie", "funsd", "fatura", "coru"):
        cfgmod.resolve_path(cfg, key).mkdir(parents=True, exist_ok=True)
    for key in ("gmail_receipts", "gmail_invoices", "gmail_legal_financial", "gmail_unclassified"):
        cfgmod.resolve_path(cfg, key).mkdir(parents=True, exist_ok=True)
    cfgmod.resolve_path(cfg, "metadata").mkdir(parents=True, exist_ok=True)


def action_str(label: str, mode: str) -> str:
    return f"{label}({mode})"


def _planned_action(mode: str) -> str:
    return f"would-{mode}"


def action_ref(label: str, ds) -> dict:
    return {"action": label, "note": f"confidence={ds.confidence}"}


if __name__ == "__main__":
    raise SystemExit(main())
