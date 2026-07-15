#!/usr/bin/env python3
"""Select public pages, render anonymized private PDF pages, and write manifests."""
from __future__ import annotations

from _rotation_cli import load, nonnegative_int, parser, print_result
from src.page_preparation import prepare_page_images


def main() -> int:
    ap = parser(__doc__)
    ap.add_argument("--dry-run", action="store_true", help="inspect and select without writing artifacts")
    ap.add_argument("--limit", type=nonnegative_int, default=0, help="debug source-file cap (0 = no cap)")
    ap.add_argument("--datasets", default=None, help="comma-separated dataset filter")
    ap.add_argument("--force", action="store_true", help="replace existing prepared private page images")
    args = ap.parse_args()
    result = prepare_page_images(
        load(args),
        dry_run=args.dry_run,
        limit=args.limit,
        datasets=args.datasets,
        force=args.force,
    )
    print_result("Page preparation", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
