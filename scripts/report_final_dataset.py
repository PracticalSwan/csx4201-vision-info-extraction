#!/usr/bin/env python3
"""Generate aggregate, public-only reports for the executed final corpus build."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config as cfgmod  # noqa: E402
from src.information_extraction.dataset_reporting import generate_final_dataset_reports  # noqa: E402
from src.information_extraction.model_dataset import profile_manifest_path  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    args = parser.parse_args()
    cfg = cfgmod.load_config(args.config)
    metadata = cfgmod.resolve_path(cfg, "metadata")
    result = generate_final_dataset_reports(
        profile_manifest_path(metadata, "final"),
        metadata / "information_extraction_split_manifest.csv",
        cfgmod.resolve_path(cfg, "reports") / "final_model",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
