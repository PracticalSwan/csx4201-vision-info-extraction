#!/usr/bin/env python3
"""Print and persist the D:-backed OCR/training environment report."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ocr.environment import DEFAULT_ASSET_ROOT, write_environment_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", default=str(DEFAULT_ASSET_ROOT))
    parser.add_argument(
        "--layout-python",
        default=str(DEFAULT_ASSET_ROOT / "environments" / "ie-layout" / "Scripts" / "python.exe"),
    )
    parser.add_argument(
        "--output", default=str(PROJECT_ROOT / "reports" / "environment" / "environment_report.json")
    )
    args = parser.parse_args()
    report = write_environment_report(args.output, args.asset_root, args.layout_python)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
