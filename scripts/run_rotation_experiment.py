#!/usr/bin/env python3
"""Run the complete page-to-rotation-to-K-Means experiment in stage order."""
from __future__ import annotations

from typing import Any, Callable

from _rotation_cli import load, parser, print_result, positive_int
from src.orientation_features import extract_rotation_features
from src.page_preparation import prepare_page_images
from src.rotation_dataset import (
    create_rotation_splits,
    generate_rotation_data,
    record_raw_baseline,
    verify_rotation_data,
)
from src.rotation_model import (
    evaluate_kmeans_rotation,
    fit_rotation_preprocessing,
    train_kmeans_rotation,
)


def main() -> int:
    ap = parser(__doc__)
    ap.add_argument("--profile", choices=("smoke", "full"), default=None)
    ap.add_argument("--force", action="store_true", help="rebuild replaceable derived artifacts")
    ap.add_argument("--workers", type=positive_int, default=None, help="worker count override")
    args = ap.parse_args()
    cfg = load(args)
    profile = args.profile or str(cfg["rotation_generation"].get("default_profile", "full"))

    stages: dict[str, Any] = {}
    stages["raw_baseline"] = _run_stage(
        "Record or load raw baseline", record_raw_baseline, cfg
    )
    stages["page_preparation"] = _run_stage(
        "Prepare page images", prepare_page_images, cfg, force=args.force
    )
    stages["split_creation"] = _run_stage("Create leakage-safe splits", create_rotation_splits, cfg)
    stages["rotation_generation"] = _run_stage(
        f"Generate {profile} rotations",
        generate_rotation_data,
        cfg,
        profile=profile,
        force=args.force,
        workers=args.workers,
    )
    first_verification = _run_stage(
        "Verify generated rotations", verify_rotation_data, cfg, profile=profile
    )
    stages["rotation_verification"] = first_verification
    if not first_verification.get("all_passed", False):
        print_result("Rotation experiment stopped after failed verification", stages)
        return 1
    stages["feature_extraction"] = _run_stage(
        "Extract orientation features",
        extract_rotation_features,
        cfg,
        profile=profile,
        force=args.force,
        workers=args.workers,
    )
    stages["preprocessing"] = _run_stage(
        "Fit train-only preprocessing", fit_rotation_preprocessing, cfg, force=args.force
    )
    stages["kmeans_training"] = _run_stage(
        "Train K-Means", train_kmeans_rotation, cfg, force=args.force
    )
    stages["mapped_evaluation"] = _run_stage(
        "Evaluate K-Means mapping", evaluate_kmeans_rotation, cfg
    )

    from src.angle_estimation import evaluate_angle_estimation

    stages["angle_evaluation"] = _run_stage(
        "Evaluate exact-angle correction",
        evaluate_angle_estimation,
        cfg,
        profile=profile,
        force=args.force,
        workers=args.workers,
    )
    final_verification = _run_stage(
        "Run final complete-pipeline verification",
        verify_rotation_data,
        cfg,
        profile=profile,
        require_model_artifacts=True,
    )
    stages["final_verification"] = final_verification
    print_result("Rotation experiment", stages)
    return 0 if final_verification.get("all_passed", False) else 1


def _run_stage(label: str, function: Callable[..., dict[str, Any]], *args: Any, **kwargs: Any) -> dict[str, Any]:
    print(f"\n== {label} ==")
    result = function(*args, **kwargs)
    summary = result.get("summary") if isinstance(result, dict) else None
    if isinstance(summary, dict):
        count = summary.get("successful_rotations", summary.get("fit_rotation_count"))
        if count is not None:
            print(f"rows/samples: {count}")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
