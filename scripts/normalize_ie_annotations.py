#!/usr/bin/env python3
"""Normalize public annotations and build leakage-safe IE manifests."""
from __future__ import annotations

from _rotation_cli import load, nonnegative_int, parser, print_result

from src.information_extraction.manifest import build_information_extraction_manifest


def main() -> int:
    ap = parser(__doc__ or "Normalize information-extraction annotations")
    ap.add_argument("--limit", type=nonnegative_int, default=0, help="debug source-row cap (0 = all)")
    ap.add_argument("--force", action="store_true", help="rewrite existing normalized JSON")
    args = ap.parse_args()
    result = build_information_extraction_manifest(load(args), limit=args.limit, force=args.force)
    print_result("annotation normalization", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
