#!/usr/bin/env python3
"""Generate smoke or bounded-full balanced rotations from the split manifest."""
from __future__ import annotations

from _rotation_cli import add_filter_arguments, load, parser, print_result
from src.rotation_dataset import generate_rotation_data


def main() -> int:
    ap = parser(__doc__)
    ap.add_argument("--profile", choices=("smoke", "full"), default="full")
    ap.add_argument("--dry-run", action="store_true", help="estimate work and disk use without writing images")
    ap.add_argument("--force", action="store_true", help="replace matching rotation image artifacts")
    add_filter_arguments(ap)
    args = ap.parse_args()
    result = generate_rotation_data(
        load(args),
        profile=args.profile,
        dry_run=args.dry_run,
        force=args.force,
        limit=args.limit,
        datasets=args.datasets,
        splits=args.splits,
        workers=args.workers,
    )
    print_result("Rotation generation", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
