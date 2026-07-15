#!/usr/bin/env python3
"""Verify rotation artifacts, split leakage, privacy, zones, and raw-data integrity."""
from __future__ import annotations

from _rotation_cli import load, parser, print_result
from src.rotation_dataset import verify_rotation_data


def main() -> int:
    ap = parser(__doc__)
    ap.add_argument("--profile", choices=("smoke", "full"), default=None)
    ap.add_argument(
        "--complete",
        action="store_true",
        help="also verify feature, preprocessing, K-Means, and exact-angle artifacts",
    )
    args = ap.parse_args()
    result = verify_rotation_data(
        load(args),
        profile=args.profile,
        require_model_artifacts=args.complete,
    )
    print_result("Rotation verification", result)
    return 0 if result.get("all_passed", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
