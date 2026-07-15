#!/usr/bin/env python3
"""Build a public-only PaddleOCR-aligned LayoutXLM dataset."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config as cfgmod  # noqa: E402
from src.information_extraction.model_dataset import prepare_model_dataset  # noqa: E402
from src.ocr.environment import configure_external_environment  # noqa: E402
from src.ocr.model_registry import ModelRegistry  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--profile", choices=("smoke", "development", "final"), default="smoke")
    parser.add_argument("--device", choices=("cpu", "gpu:0"), default="cpu")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--model-setup", default=str(PROJECT_ROOT / "reports" / "ocr" / "model_setup.json"))
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    cfg = cfgmod.load_config(args.config)
    configure_external_environment(cfgmod.resolve_path(cfg, "external_assets"))
    registry = ModelRegistry.from_setup(args.model_setup)
    summary = prepare_model_dataset(
        cfg, registry, profile=args.profile, device=args.device, limit=args.limit, force=args.force
    )
    import json

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
