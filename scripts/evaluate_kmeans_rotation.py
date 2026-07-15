#!/usr/bin/env python3
"""Evaluate raw clusters and the train-learned cluster-to-zone mapping."""
from __future__ import annotations

from _rotation_cli import load, parser, print_result
from src.rotation_model import evaluate_kmeans_rotation


def main() -> int:
    ap = parser(__doc__)
    args = ap.parse_args()
    result = evaluate_kmeans_rotation(load(args))
    print_result("Mapped K-Means evaluation", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
