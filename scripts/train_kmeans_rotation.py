#!/usr/bin/env python3
"""Fit four-cluster K-Means and learn the train-only Hungarian zone mapping."""
from __future__ import annotations

from _rotation_cli import load, parser, print_result
from src.rotation_model import train_kmeans_rotation


def main() -> int:
    ap = parser(__doc__)
    ap.add_argument("--force", action="store_true", help="refit K-Means and its cluster-zone mapping")
    args = ap.parse_args()
    result = train_kmeans_rotation(load(args), force=args.force)
    print_result("K-Means rotation training", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
