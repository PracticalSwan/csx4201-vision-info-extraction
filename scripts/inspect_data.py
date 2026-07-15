#!/usr/bin/env python3
"""inspect_data.py — print the discovered dataset tree and file-type counts.

Read-only. Nothing is moved or written.

Usage:
    python scripts/inspect_data.py [--config config.yaml] [--depth 3] [--log-level INFO]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

# Force UTF-8 output so non-ASCII filenames (e.g. Thai) render on Windows console.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config as cfgmod  # noqa: E402
from src import dataset_discovery as dd  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--log-level", default=None)
    args = ap.parse_args()

    cfg = cfgmod.load_config(args.config)
    cfgmod.setup_logging(cfg, args.log_level)

    print(f"Project root: {cfgmod.project_root(cfg)}")
    print(f"Candidate roots: {[str(cfgmod.resolve_path(cfg, r)) for r in cfg['discovery']['candidate_roots']]}")
    print()

    datasets = dd.discover_datasets(cfg)
    if not datasets:
        print("No datasets discovered.")
        return 1

    for ds in datasets:
        print("=" * 70)
        print(f"Dataset: {ds.name}  ({ds.source_type})  confidence={ds.confidence}")
        print(f"  current: {ds.current_path}")
        print(f"  target : {ds.target_path}")
        print(f"  evidence: {', '.join(ds.evidence) if ds.evidence else '(none)'}")
        if ds.current_path.exists():
            print_tree(ds.current_path, max_depth=args.depth)
            counts = count_extensions(ds.current_path)
            total = sum(counts.values())
            size = sum(p.stat().st_size for p in ds.current_path.rglob("*") if p.is_file())
            print(f"  files: {total}   size: {human_size(size)}")
            print(f"  extensions: {dict(counts.most_common())}")
        else:
            print("  (current path does not exist)")
        print()

    return 0


def print_tree(root: Path, max_depth: int, prefix: str = "", depth: int = 0) -> None:
    if depth > max_depth:
        return
    try:
        entries = sorted(root.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except (PermissionError, OSError):
        return
    if depth == 0:
        print(f"  {root.name}/")
    dirs = [e for e in entries if e.is_dir()]
    files_count = sum(1 for e in entries if e.is_file())
    for i, d in enumerate(dirs):
        last = i == len(dirs) - 1 and files_count == 0
        branch = "└── " if last else "├── "
        print(f"  {prefix}{branch}{d.name}/")
        print_tree(d, max_depth, prefix + ("    " if last else "│   "), depth + 1)
    if files_count:
        print(f"  {prefix}└── ({files_count} files)")


def count_extensions(root: Path) -> Counter:
    c: Counter = Counter()
    try:
        for p in root.rglob("*"):
            if p.is_file():
                c[p.suffix.lower() or "<none>"] += 1
    except (PermissionError, OSError):
        pass
    return c


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


if __name__ == "__main__":
    raise SystemExit(main())
