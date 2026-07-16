#!/usr/bin/env python3
"""Fit temperatures and abstention thresholds on public dev_calibration only."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config as cfgmod  # noqa: E402
from src.information_extraction.layoutxlm_data import BIO_LABELS, load_model_examples  # noqa: E402
from src.information_extraction.model_dataset import (  # noqa: E402
    profile_manifest_path,
    validate_manifest_profile,
)
from src.information_extraction.multitask_calibration import (  # noqa: E402
    choose_document_threshold,
    choose_positive_threshold,
    expected_calibration_error,
    positive_f1_at_threshold,
)
from src.information_extraction.multitask_data import (  # noqa: E402
    CANONICAL_FIELD_LABELS,
    DOCUMENT_TYPE_LABELS,
    RELATION_LABELS,
)
from src.information_extraction.multitask_evaluation import validate_evaluation_binding  # noqa: E402
from src.ocr.environment import configure_external_environment, require_storage_gate  # noqa: E402
from src.rotation_common import atomic_write_json, configuration_hash, read_csv_rows  # noqa: E402
from scripts.train_multitask_model import (  # noqa: E402
    TokenizedWindowDataset,
    _collate,
    _device,
    _seed_everything,
    _sha256,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--profile", choices=("development", "final"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--streams",
        nargs="+",
        choices=("ground_truth", "paddleocr"),
        default=("ground_truth", "paddleocr"),
        help="Hybrid is intentionally excluded because it is training-only.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "models" / "multitask_calibration.json"),
    )
    args = parser.parse_args()

    cfg = cfgmod.load_config(args.config)
    asset_root = cfgmod.resolve_path(cfg, "external_assets")
    configure_external_environment(asset_root)
    require_storage_gate(
        asset_root,
        operation=f"{args.profile} calibration",
        anticipated_c_gib=0.25,
        anticipated_asset_gib=5.0,
    )

    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import LayoutXLMTokenizerFast

    from src.information_extraction.layoutxlm_model import MultiTaskTextLayoutModel

    seed = int(cfg.get("layout_model", {}).get("seed", 42))
    _seed_everything(seed, torch)
    selected_device = _device(args.device, torch)
    checkpoint = Path(args.checkpoint).resolve()
    state_path = checkpoint / "training_state.json"
    model_path = checkpoint / "model.safetensors"
    if not state_path.is_file() or not model_path.is_file():
        raise SystemExit(f"checkpoint is incomplete: {checkpoint}")
    training_state = json.loads(state_path.read_text(encoding="utf-8"))

    manifest_path = profile_manifest_path(
        cfgmod.resolve_path(cfg, "metadata"), args.profile
    )
    manifest_rows = read_csv_rows(manifest_path)
    build_ids = {row.get("build_id", "") for row in manifest_rows}
    if len(build_ids) != 1 or "" in build_ids:
        raise SystemExit(f"manifest build IDs are missing or mixed: {sorted(build_ids)!r}")
    build_id = next(iter(build_ids))
    validate_manifest_profile(
        manifest_rows,
        expected_profile=args.profile,
        expected_build_id=build_id,
    )
    try:
        validate_evaluation_binding(
            training_state,
            expected_profile=args.profile,
            evaluation_build_id=build_id,
            allow_cross_build=False,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    token_sources = set(args.streams)
    examples = load_model_examples(
        manifest_path,
        "dev_calibration",
        expected_profile=args.profile,
        expected_build_id=build_id,
        token_sources=token_sources,
    )
    if not examples:
        raise SystemExit("dev_calibration has no examples for the requested streams")
    if any(example.get("is_private") is not False for example in examples):
        raise SystemExit("private or unmarked calibration example detected")

    tokenizer = LayoutXLMTokenizerFast.from_pretrained(checkpoint)
    model = MultiTaskTextLayoutModel.from_pretrained(checkpoint).to(selected_device).eval()
    cache_signature = configuration_hash({
        "schema": "multitask-window-v1",
        "build_id": build_id,
        "max_length": args.max_length,
        "stride": 64,
        "tokenizer": "microsoft/layoutxlm-base",
        "entity_labels": BIO_LABELS,
        "canonical_labels": CANONICAL_FIELD_LABELS,
        "relation_labels": RELATION_LABELS,
    })
    dataset = TokenizedWindowDataset(
        examples,
        tokenizer,
        cfgmod.resolve_path(cfg, "model_datasets")
        / args.profile
        / build_id
        / "tokenized"
        / cache_signature
        / "dev_calibration",
        max_length=args.max_length,
        seed=seed,
        torch_module=torch,
        dataset_base=Dataset,
    )
    tensors = _collect_logits(
        model,
        dataset,
        selected_device,
        torch,
        DataLoader,
    )
    tensors = {
        task: {
            "logits": values["logits"].to(selected_device),
            "labels": values["labels"].to(selected_device),
        }
        for task, values in tensors.items()
    }
    temperatures = {
        task: _fit_temperature(values["logits"], values["labels"], torch)
        for task, values in tensors.items()
    }
    calibrated = {
        task: torch.softmax(values["logits"] / temperatures[task], dim=-1)
        for task, values in tensors.items()
    }
    thresholds: dict[str, float] = {}
    metrics: dict[str, Any] = {}
    for task in ("entity", "canonical", "relation"):
        probabilities = calibrated[task]
        labels = tensors[task]["labels"]
        predictions = probabilities.argmax(dim=-1)
        confidence = probabilities.max(dim=-1).values
        threshold = choose_positive_threshold(
            confidence.tolist(), predictions.tolist(), labels.tolist()
        )
        thresholds[task] = threshold
        metrics[task] = _task_metrics(
            tensors[task]["logits"],
            labels,
            probabilities,
            temperature=temperatures[task],
            threshold=threshold,
            torch=torch,
        )

    document_probabilities = calibrated["document"]
    document_labels = tensors["document"]["labels"]
    document_predictions = document_probabilities.argmax(dim=-1)
    document_confidence = document_probabilities.max(dim=-1).values
    document_correct = document_predictions.eq(document_labels)
    thresholds["document"] = choose_document_threshold(
        document_confidence.tolist(),
        document_correct.tolist(),
        minimum_coverage=0.90,
    )
    metrics["document"] = _document_metrics(
        tensors["document"]["logits"],
        document_labels,
        document_probabilities,
        temperature=temperatures["document"],
        threshold=thresholds["document"],
        torch=torch,
    )

    canonical_probabilities = calibrated["canonical"]
    canonical_predictions = canonical_probabilities.argmax(dim=-1)
    canonical_labels = tensors["canonical"]["labels"]
    canonical_field_thresholds = {}
    for label_id, field in enumerate(CANONICAL_FIELD_LABELS[1:], start=1):
        support = int(canonical_labels.eq(label_id).sum())
        if support < 5:
            continue
        field_predictions = [
            label_id if int(value) == label_id else 0
            for value in canonical_predictions.tolist()
        ]
        field_labels = [
            label_id if int(value) == label_id else 0
            for value in canonical_labels.tolist()
        ]
        canonical_field_thresholds[field] = choose_positive_threshold(
            canonical_probabilities[:, label_id].tolist(),
            field_predictions,
            field_labels,
        )

    report = {
        "schema_version": "1.0",
        "profile": args.profile,
        "split": "dev_calibration",
        "token_sources": sorted(token_sources),
        "public_only": True,
        "private_example_count": 0,
        "gmail_fit_rows": 0,
        "checkpoint": str(checkpoint),
        "checkpoint_model_sha256": _sha256(model_path),
        "checkpoint_build_id": str(training_state.get("build_id", "")),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "example_count": len(examples),
        "window_count": len(dataset),
        "temperatures": temperatures,
        "thresholds": thresholds,
        "canonical_field_thresholds": canonical_field_thresholds,
        "metrics": metrics,
        "label_support": {
            "entity": dict(Counter(tensors["entity"]["labels"].tolist())),
            "canonical": dict(Counter(tensors["canonical"]["labels"].tolist())),
            "document": dict(Counter(tensors["document"]["labels"].tolist())),
            "relation": dict(Counter(tensors["relation"]["labels"].tolist())),
        },
        "device": str(selected_device),
    }
    output_path = Path(args.output).resolve()
    atomic_write_json(output_path, report)
    report_path = (
        cfgmod.resolve_path(cfg, "reports")
        / "final_model"
        / f"calibration_{args.profile}.json"
    )
    atomic_write_json(report_path, report)
    print(json.dumps({
        "output": str(output_path),
        "report": str(report_path),
        "profile": args.profile,
        "build_id": build_id,
        "examples": len(examples),
        "windows": len(dataset),
        "temperatures": temperatures,
        "thresholds": thresholds,
        "canonical_field_thresholds": canonical_field_thresholds,
    }, indent=2))
    return 0


def _collect_logits(model: Any, dataset: Any, device: Any, torch: Any, data_loader: Any) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, list[Any]]] = {
        task: {"logits": [], "labels": []}
        for task in ("entity", "canonical", "document", "relation")
    }
    loader = data_loader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda items: _collate(items, torch),
    )
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            for task, logits_name, labels_name in (
                ("entity", "entity_logits", "entity_labels"),
                ("canonical", "canonical_logits", "canonical_labels"),
                ("document", "document_logits", "document_labels"),
            ):
                logits = getattr(output, logits_name).reshape(-1, getattr(output, logits_name).shape[-1])
                labels = batch[labels_name].reshape(-1)
                valid = labels.ne(-100)
                values[task]["logits"].append(logits[valid].detach().cpu())
                values[task]["labels"].append(labels[valid].detach().cpu())
            if output.relation_logits is not None and "relation_labels" in batch:
                logits = output.relation_logits.reshape(-1, output.relation_logits.shape[-1])
                labels = batch["relation_labels"].reshape(-1)
                valid = labels.ne(-100)
                values["relation"]["logits"].append(logits[valid].detach().cpu())
                values["relation"]["labels"].append(labels[valid].detach().cpu())
    result = {}
    for task, parts in values.items():
        if not parts["logits"]:
            raise RuntimeError(f"dev_calibration has no {task} targets")
        result[task] = {
            "logits": torch.cat(parts["logits"], dim=0),
            "labels": torch.cat(parts["labels"], dim=0),
        }
    return result


def _fit_temperature(logits: Any, labels: Any, torch: Any) -> float:
    """Bounded two-stage grid search avoids a costly gradient loop on token logits."""
    coarse = torch.logspace(
        math.log10(0.25),
        math.log10(4.0),
        steps=21,
        dtype=torch.float32,
        device=logits.device,
    )
    coarse_losses = _temperature_nlls(logits, labels, coarse, torch)
    best_index = int(coarse_losses.argmin())
    lower = coarse[max(0, best_index - 1)]
    upper = coarse[min(len(coarse) - 1, best_index + 1)]
    refined = torch.linspace(
        float(lower), float(upper), steps=11, device=logits.device
    )
    refined_losses = _temperature_nlls(logits, labels, refined, torch)
    return float(refined[int(refined_losses.argmin())])


def _temperature_nlls(logits: Any, labels: Any, temperatures: Any, torch: Any) -> Any:
    totals = torch.zeros(
        len(temperatures), dtype=torch.float64, device=logits.device
    )
    count = 0
    values = logits.to(torch.float32)
    for start in range(0, len(values), 20_000):
        chunk = values[start : start + 20_000]
        targets = labels[start : start + 20_000]
        scaled = chunk.unsqueeze(0) / temperatures[:, None, None]
        log_probabilities = torch.log_softmax(scaled, dim=-1)
        selected = log_probabilities.gather(
            2,
            targets[None, :, None].expand(len(temperatures), -1, 1),
        ).squeeze(-1)
        totals -= selected.sum(dim=1).to(torch.float64)
        count += len(chunk)
    return totals / max(1, count)


def _task_metrics(logits: Any, labels: Any, probabilities: Any, *, temperature: float, threshold: float, torch: Any) -> dict[str, Any]:
    predictions = probabilities.argmax(dim=-1)
    confidence = probabilities.max(dim=-1).values
    correct = predictions.eq(labels)
    before_probabilities = torch.softmax(logits, dim=-1)
    before_confidence = before_probabilities.max(dim=-1).values
    return {
        "support": int(labels.numel()),
        "temperature": temperature,
        "threshold": threshold,
        "nll_before": float(torch.nn.functional.cross_entropy(logits, labels)),
        "nll_after": float(torch.nn.functional.cross_entropy(logits / temperature, labels)),
        "ece_before": expected_calibration_error(before_confidence.tolist(), before_probabilities.argmax(dim=-1).eq(labels).tolist()),
        "ece_after": expected_calibration_error(confidence.tolist(), correct.tolist()),
        "positive_f1_without_abstention": positive_f1_at_threshold(confidence.tolist(), predictions.tolist(), labels.tolist(), threshold=0.0),
        "positive_f1_with_abstention": positive_f1_at_threshold(confidence.tolist(), predictions.tolist(), labels.tolist(), threshold=threshold),
    }


def _document_metrics(logits: Any, labels: Any, probabilities: Any, *, temperature: float, threshold: float, torch: Any) -> dict[str, Any]:
    predictions = probabilities.argmax(dim=-1)
    confidence = probabilities.max(dim=-1).values
    correct = predictions.eq(labels)
    retained = confidence.ge(threshold)
    before_probabilities = torch.softmax(logits, dim=-1)
    return {
        "support": int(labels.numel()),
        "temperature": temperature,
        "threshold": threshold,
        "nll_before": float(torch.nn.functional.cross_entropy(logits, labels)),
        "nll_after": float(torch.nn.functional.cross_entropy(logits / temperature, labels)),
        "ece_before": expected_calibration_error(before_probabilities.max(dim=-1).values.tolist(), before_probabilities.argmax(dim=-1).eq(labels).tolist()),
        "ece_after": expected_calibration_error(confidence.tolist(), correct.tolist()),
        "coverage": float(retained.float().mean()),
        "selective_accuracy": float(correct[retained].float().mean()) if bool(retained.any()) else None,
        "class_count": len(DOCUMENT_TYPE_LABELS),
    }


if __name__ == "__main__":
    raise SystemExit(main())
