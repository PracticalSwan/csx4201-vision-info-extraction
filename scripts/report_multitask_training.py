#!/usr/bin/env python3
"""Materialize compact final training history and bounded trial evidence."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config as cfgmod  # noqa: E402
from src.rotation_common import atomic_write_csv, atomic_write_json  # noqa: E402


TRIALS = (
    {
        "trial": "trial_1_ground_truth",
        "streams": "ground_truth",
        "upright_probability": 1.0,
        "checkpoint_selection": "upright dev_select",
        "notes": "ground-truth baseline",
    },
    {
        "trial": "trial_2_rotation_variants",
        "streams": "ground_truth,paddleocr,hybrid",
        "upright_probability": 1.0,
        "checkpoint_selection": "upright dev_select",
        "notes": "real OCR-noise and hybrid target streams",
    },
    {
        "trial": "trial_3_dynamic_rotation",
        "streams": "ground_truth,paddleocr,hybrid",
        "upright_probability": 0.2,
        "checkpoint_selection": "upright dev_select",
        "notes": "80 percent arbitrary-angle augmentation",
    },
    {
        "trial": "trial_4_balanced_rotation",
        "streams": "ground_truth,paddleocr,hybrid",
        "upright_probability": 0.6,
        "checkpoint_selection": "upright dev_select",
        "notes": "selected 60/40 upright/arbitrary robustness compromise",
    },
)

TRIAL_COLUMNS = (
    "trial", "streams", "upright_probability", "rotation_probability",
    "clean_entity_f1", "clean_canonical_f1", "clean_relation_f1", "clean_composite",
    "ocr_variant_composite", "rotated_37_composite", "three_gate_mean_composite",
    "checkpoint_selection", "selected_for_final", "notes",
)

HISTORY_COLUMNS = (
    "epoch", "loss", "entity_token_f1", "canonical_evidence_token_f1",
    "relation_f1", "document_macro_f1", "upright_composite_score",
    "rotated_37_entity_token_f1", "rotated_37_canonical_evidence_token_f1",
    "rotated_37_relation_f1", "rotated_37_document_macro_f1",
    "rotated_37_composite_score", "selection_composite_score",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    args = parser.parse_args()
    cfg = cfgmod.load_config(args.config)
    report_root = cfgmod.resolve_path(cfg, "reports") / "final_model"
    evaluation_root = report_root / "evaluations"
    trial_rows = [_trial_row(definition, evaluation_root) for definition in TRIALS]
    atomic_write_csv(report_root / "hyperparameter_trials.csv", trial_rows, TRIAL_COLUMNS)

    training_path = report_root / "multitask_training_final.json"
    if not training_path.is_file():
        raise FileNotFoundError("final training report is not available yet")
    training = json.loads(training_path.read_text(encoding="utf-8"))
    history_rows = [_history_row(item) for item in training.get("validation_history", [])]
    atomic_write_csv(report_root / "training_history.csv", history_rows, HISTORY_COLUMNS)
    summary = {
        key: training.get(key)
        for key in (
            "schema_version", "profile", "build_id", "manifest_path", "manifest_sha256",
            "public_only", "gmail_fit_rows", "source_checkpoint", "architecture",
            "visual_backbone", "license", "device",
            "mixed_precision", "token_sources", "train_examples", "validation_examples",
            "train_windows", "validation_windows", "rotated_validation_windows",
            "training_target_counts", "dynamic_rotation", "mean_task_losses",
            "optimizer_steps", "micro_steps", "requested_epochs", "completed_epochs",
            "early_stopping_patience", "stopped_early", "stop_reason", "best_epoch",
            "best_composite_score", "checkpoint_selection", "validation", "checkpoint",
            "checkpoint_reload_passed", "checkpoint_reload_max_difference", "duration_seconds",
            "limitations",
        )
    }
    summary["bounded_development_trial_count"] = len(trial_rows)
    summary["selected_development_trial"] = "trial_4_balanced_rotation"
    atomic_write_json(report_root / "training_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


def _trial_row(definition: dict[str, Any], evaluation_root: Path) -> dict[str, Any]:
    number = definition["trial"].split("_", 2)[1]
    prefix = f"trial_{number}_dev_select"
    clean = _metrics(evaluation_root / f"{prefix}_ground_truth.json")
    variants = _metrics(evaluation_root / f"{prefix}_variants.json")
    rotated = _metrics(evaluation_root / f"{prefix}_ground_truth_rotated_37.json")
    composites = [
        item["composite_score"] for item in (clean, variants, rotated) if item
    ]
    return {
        **definition,
        "rotation_probability": 1.0 - float(definition["upright_probability"]),
        "clean_entity_f1": clean.get("entity_token_f1"),
        "clean_canonical_f1": clean.get("canonical_evidence_token_f1"),
        "clean_relation_f1": clean.get("relation_f1"),
        "clean_composite": clean.get("composite_score"),
        "ocr_variant_composite": variants.get("composite_score"),
        "rotated_37_composite": rotated.get("composite_score"),
        "three_gate_mean_composite": statistics.fmean(composites) if composites else None,
        "selected_for_final": definition["trial"] == "trial_4_balanced_rotation",
    }


def _metrics(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return dict(json.loads(path.read_text(encoding="utf-8")).get("metrics") or {})


def _history_row(item: dict[str, Any]) -> dict[str, Any]:
    rotated = dict(item.get("rotated_37") or {})
    return {
        "epoch": item.get("epoch"),
        "loss": item.get("loss"),
        "entity_token_f1": item.get("entity_token_f1"),
        "canonical_evidence_token_f1": item.get("canonical_evidence_token_f1"),
        "relation_f1": item.get("relation_f1"),
        "document_macro_f1": item.get("document_macro_f1"),
        "upright_composite_score": item.get("composite_score"),
        "rotated_37_entity_token_f1": rotated.get("entity_token_f1"),
        "rotated_37_canonical_evidence_token_f1": rotated.get("canonical_evidence_token_f1"),
        "rotated_37_relation_f1": rotated.get("relation_f1"),
        "rotated_37_document_macro_f1": rotated.get("document_macro_f1"),
        "rotated_37_composite_score": rotated.get("composite_score"),
        "selection_composite_score": item.get("selection_composite_score"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
