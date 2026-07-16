#!/usr/bin/env python3
"""Build profile-bound public ground-truth, PaddleOCR, and hybrid streams."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config as cfgmod  # noqa: E402
from src.information_extraction.model_dataset import prepare_model_dataset  # noqa: E402
from src.ocr.environment import configure_external_environment, require_storage_gate  # noqa: E402
from src.ocr.model_registry import ModelRegistry  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--profile", choices=("smoke", "development", "final"), default="smoke")
    parser.add_argument("--device", choices=("cpu", "gpu:0"), default="cpu")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--ocr-variant-limit",
        type=int,
        default=0,
        help="Bound only PaddleOCR/hybrid variants; ground-truth pages remain complete.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--streams",
        nargs="+",
        choices=("ground_truth", "paddleocr", "hybrid"),
        default=("ground_truth",),
    )
    parser.add_argument("--model-setup", default=str(PROJECT_ROOT / "reports" / "ocr" / "model_setup.json"))
    args = parser.parse_args()
    if args.limit < 0 or args.ocr_variant_limit < 0:
        parser.error("--limit and --ocr-variant-limit must be non-negative")
    cfg = cfgmod.load_config(args.config)
    asset_root = cfgmod.resolve_path(cfg, "external_assets")
    configure_external_environment(asset_root)
    anticipated_asset_gib = {"smoke": 1.0, "development": 5.0, "final": 15.0}[args.profile]
    require_storage_gate(
        asset_root,
        operation=f"{args.profile} model-dataset preparation",
        anticipated_c_gib=0.25,
        anticipated_asset_gib=anticipated_asset_gib,
    )
    registry = ModelRegistry.from_setup(args.model_setup)
    summary = prepare_model_dataset(
        cfg,
        registry,
        profile=args.profile,
        device=args.device,
        limit=args.limit,
        force=args.force,
        streams=tuple(args.streams),
        ocr_variant_limit=args.ocr_variant_limit,
    )
    import json

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
