#!/usr/bin/env python3
"""Evaluate exact-angle estimation and correction on generated rotations."""
from __future__ import annotations

from _rotation_cli import add_filter_arguments, load, parser, print_result


def main() -> int:
    ap = parser(__doc__)
    ap.add_argument("--profile", choices=("smoke", "full"), default=None)
    ap.add_argument("--force", action="store_true", help="replace existing angle-evaluation artifacts")
    add_filter_arguments(ap)
    args = ap.parse_args()

    # Kept local so --help remains available while the optional stage module is
    # being installed or developed independently.
    from src.angle_estimation import evaluate_angle_estimation

    result = evaluate_angle_estimation(
        load(args),
        profile=args.profile,
        force=args.force,
        limit=args.limit,
        datasets=args.datasets,
        splits=args.splits,
        workers=args.workers,
    )
    print_result("Exact-angle evaluation", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
