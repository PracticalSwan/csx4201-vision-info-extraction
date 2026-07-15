#!/usr/bin/env python3
"""Create deterministic, leakage-safe public splits and an isolated private-test split."""
from __future__ import annotations

from _rotation_cli import load, parser, print_result
from src.rotation_dataset import create_rotation_splits


def main() -> int:
    ap = parser(__doc__)
    ap.add_argument("--dry-run", action="store_true", help="validate assignments without writing split files")
    ap.add_argument("--seed", type=int, default=None, help="override the configured split seed")
    args = ap.parse_args()
    result = create_rotation_splits(load(args), dry_run=args.dry_run, seed=args.seed)
    print_result("Rotation split creation", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
