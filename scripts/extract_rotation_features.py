#!/usr/bin/env python3
"""Extract deterministic HOG, Hough, projection, edge, and geometry features."""
from __future__ import annotations

from _rotation_cli import add_filter_arguments, load, parser, print_result
from src.orientation_features import extract_rotation_features


def main() -> int:
    ap = parser(__doc__)
    ap.add_argument("--profile", choices=("smoke", "full"), default=None)
    ap.add_argument("--force", action="store_true", help="rebuild feature caches")
    add_filter_arguments(ap)
    args = ap.parse_args()
    result = extract_rotation_features(
        load(args),
        profile=args.profile,
        force=args.force,
        limit=args.limit,
        datasets=args.datasets,
        splits=args.splits,
        workers=args.workers,
    )
    print_result("Rotation feature extraction", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
