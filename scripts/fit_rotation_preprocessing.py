#!/usr/bin/env python3
"""Fit the train-only scaler/PCA pipeline and transform every available split."""
from __future__ import annotations

from _rotation_cli import load, parser, print_result
from src.rotation_model import fit_rotation_preprocessing


def main() -> int:
    ap = parser(__doc__)
    ap.add_argument("--force", action="store_true", help="refit preprocessing artifacts")
    args = ap.parse_args()
    result = fit_rotation_preprocessing(load(args), force=args.force)
    print_result("Rotation preprocessing", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
