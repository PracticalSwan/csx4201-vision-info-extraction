#!/usr/bin/env python3
"""Train, resume, select, save, and reload the real public multi-task pre-model."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config as cfgmod  # noqa: E402
from src.information_extraction.layoutxlm_data import (  # noqa: E402
    BIO_LABELS,
    ID_TO_LABEL,
    LABEL_TO_ID,
    load_model_examples,
    normalize_bbox,
    rotate_example_geometry,
    rotated_word_boxes,
)
from src.information_extraction.geometry import DynamicRotation  # noqa: E402
from src.information_extraction.model_dataset import (  # noqa: E402
    profile_manifest_path,
    validate_manifest_profile,
)
from src.information_extraction.multitask_data import (  # noqa: E402
    CANONICAL_FIELD_LABELS,
    DOCUMENT_TYPE_LABELS,
    ENTITY_TYPE_LABELS,
    RELATION_LABELS,
    encode_multitask_windows,
)
from src.information_extraction.multitask_calibration import apply_abstention  # noqa: E402
from src.ocr.environment import configure_external_environment, require_storage_gate  # noqa: E402
from src.rotation_common import (  # noqa: E402
    atomic_write_json,
    configuration_hash,
    read_csv_rows,
)

PROFILE_DEFAULTS = {
    "smoke": {"epochs": 1, "max_optimizer_steps": 4, "max_length": 128, "top_layers": 0},
    "development": {"epochs": 3, "max_optimizer_steps": 0, "max_length": 512, "top_layers": 4},
    "final": {"epochs": 4, "max_optimizer_steps": 0, "max_length": 512, "top_layers": 6},
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--profile", choices=tuple(PROFILE_DEFAULTS), default="smoke")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-steps", type=int, default=None, help="maximum optimizer steps")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="validation epochs without improvement before stopping (final default: 2)",
    )
    parser.add_argument("--resume", default=None, help="resume directory created by this script")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--tiny-overfit", action="store_true")
    parser.add_argument(
        "--upright-probability",
        type=float,
        default=None,
        help="Override the dynamic-rotation upright fraction for a bounded trial.",
    )
    parser.add_argument(
        "--streams",
        nargs="+",
        choices=("ground_truth", "paddleocr", "hybrid"),
        default=("ground_truth",),
    )
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.early_stopping_patience is not None and args.early_stopping_patience < 0:
        parser.error("--early-stopping-patience must be non-negative")
    if args.upright_probability is not None and not 0.0 <= args.upright_probability <= 1.0:
        parser.error("--upright-probability must be in [0, 1]")

    cfg = cfgmod.load_config(args.config)
    asset_root = cfgmod.resolve_path(cfg, "external_assets")
    configure_external_environment(asset_root)
    anticipated_asset_gib = {"smoke": 2.0, "development": 10.0, "final": 30.0}[args.profile]
    require_storage_gate(
        asset_root,
        operation=f"{args.profile} multi-task training",
        anticipated_c_gib=0.5,
        anticipated_asset_gib=anticipated_asset_gib,
    )

    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoConfig, LayoutXLMTokenizerFast, get_linear_schedule_with_warmup

    from src.information_extraction.layoutxlm_model import MultiTaskTextLayoutModel

    settings = dict(PROFILE_DEFAULTS[args.profile])
    if args.max_steps is not None:
        settings["max_optimizer_steps"] = args.max_steps
    if args.epochs is not None:
        settings["epochs"] = args.epochs
    if args.tiny_overfit:
        settings.update({"epochs": args.epochs or 20, "max_optimizer_steps": args.max_steps or 200, "max_length": 256, "top_layers": 2})
    early_stopping_patience = (
        int(args.early_stopping_patience)
        if args.early_stopping_patience is not None
        else (2 if args.profile == "final" and not args.tiny_overfit else 0)
    )
    seed = int(cfg.get("layout_model", {}).get("seed", 42))
    _seed_everything(seed, torch)
    selected_device = _device(args.device, torch)

    manifest_path = profile_manifest_path(cfgmod.resolve_path(cfg, "metadata"), args.profile)
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
    token_sources = set(args.streams)
    train_examples = load_model_examples(
        manifest_path,
        "train",
        expected_profile=args.profile,
        expected_build_id=build_id,
        token_sources=token_sources,
    )
    validation_examples = load_model_examples(
        manifest_path,
        "dev_select",
        expected_profile=args.profile,
        expected_build_id=build_id,
        token_sources=token_sources,
    )
    if not train_examples:
        raise SystemExit("no usable public training examples; build this exact profile first")
    if not validation_examples:
        raise SystemExit("no usable public dev_select examples; split/profile build is invalid")
    if args.tiny_overfit:
        train_examples = _tiny_examples(train_examples, limit=16)
        validation_examples = train_examples
    if any(example.get("is_private") is not False for example in train_examples + validation_examples):
        raise SystemExit("private or unmarked model example detected; training refused")

    checkpoint_id = str(cfg.get("layout_model", {}).get("checkpoint", "microsoft/layoutxlm-base"))
    source_checkpoint = args.resume or checkpoint_id
    tokenizer = LayoutXLMTokenizerFast.from_pretrained(
        source_checkpoint,
        cache_dir=str(cfgmod.resolve_path(cfg, "layout_models")),
    )
    model_config = AutoConfig.from_pretrained(
        source_checkpoint,
        cache_dir=str(cfgmod.resolve_path(cfg, "layout_models")),
    )
    model_config.num_labels = len(BIO_LABELS)
    model_config.id2label = ID_TO_LABEL
    model_config.label2id = LABEL_TO_ID
    model_config.num_document_labels = len(DOCUMENT_TYPE_LABELS)
    model_config.num_canonical_labels = len(CANONICAL_FIELD_LABELS)
    model_config.num_relation_labels = len(RELATION_LABELS)
    model_config.num_entity_types = len(ENTITY_TYPE_LABELS)
    model_config.relation_geometry_size = 10
    model_config.document_labels = list(DOCUMENT_TYPE_LABELS)
    model_config.canonical_field_labels = list(CANONICAL_FIELD_LABELS)
    model_config.relation_labels = list(RELATION_LABELS)
    model_config.entity_type_labels = list(ENTITY_TYPE_LABELS)
    model_config.visual_backbone = False
    model_config.architecture_description = "multilingual text plus normalized 2D layout"
    model = MultiTaskTextLayoutModel.from_pretrained(
        source_checkpoint,
        config=model_config,
        cache_dir=str(cfgmod.resolve_path(cfg, "layout_models")),
        ignore_mismatched_sizes=not bool(args.resume),
    )
    model.to(selected_device)

    upright_probability = (
        float(args.upright_probability)
        if args.upright_probability is not None
        else float(cfg.get("augmentation", {}).get("upright_probability", 0.2))
    )
    cache_signature = configuration_hash({
        "schema": "multitask-window-v1",
        "build_id": build_id,
        "max_length": settings["max_length"],
        "stride": 64,
        "tokenizer": checkpoint_id,
        "entity_labels": BIO_LABELS,
        "canonical_labels": CANONICAL_FIELD_LABELS,
        "relation_labels": RELATION_LABELS,
        "dynamic_rotation": {
            "enabled": not args.tiny_overfit and args.profile in {"development", "final"},
            "upright_probability": upright_probability,
            "angle_min": float(cfg.get("augmentation", {}).get("angle_min", 0.0)),
            "angle_max": float(cfg.get("augmentation", {}).get("angle_max", 360.0)),
        },
    })
    tokenized_root = (
        cfgmod.resolve_path(cfg, "model_datasets")
        / args.profile
        / build_id
        / "tokenized"
        / cache_signature
    )
    dynamic_rotation = None
    if not args.tiny_overfit and args.profile in {"development", "final"}:
        dynamic_rotation = DynamicRotation(
            seed=seed,
            upright_probability=upright_probability,
            angle_min=float(cfg.get("augmentation", {}).get("angle_min", 0.0)),
            angle_max=float(cfg.get("augmentation", {}).get("angle_max", 360.0)),
        )
    train_dataset = TokenizedWindowDataset(
        train_examples,
        tokenizer,
        tokenized_root / "train",
        max_length=int(settings["max_length"]),
        seed=seed,
        torch_module=torch,
        dataset_base=Dataset,
        dynamic_rotation=dynamic_rotation,
    )
    validation_dataset = TokenizedWindowDataset(
        validation_examples,
        tokenizer,
        tokenized_root / ("tiny" if args.tiny_overfit else "dev_select"),
        max_length=int(settings["max_length"]),
        seed=seed,
        torch_module=torch,
        dataset_base=Dataset,
    )
    rotated_validation_angle = 37.0 if args.profile == "final" and not args.tiny_overfit else None
    rotated_validation_dataset = (
        TokenizedWindowDataset(
            validation_examples,
            tokenizer,
            tokenized_root / "dev_select_rotated_37",
            max_length=int(settings["max_length"]),
            seed=seed,
            torch_module=torch,
            dataset_base=Dataset,
            fixed_rotation_angle=rotated_validation_angle,
        )
        if rotated_validation_angle is not None
        else None
    )
    class_weights = {
        "entity": _inverse_sqrt_weights(
            train_dataset.label_counts["entity"], len(BIO_LABELS)
        ),
        "document": _inverse_sqrt_weights(
            train_dataset.label_counts["document"], len(DOCUMENT_TYPE_LABELS)
        ),
        "canonical": _inverse_sqrt_weights(
            train_dataset.label_counts["canonical"], len(CANONICAL_FIELD_LABELS)
        ),
        "relation": _inverse_sqrt_weights(
            train_dataset.label_counts["relation"], len(RELATION_LABELS)
        ),
    }
    model.config.entity_class_weights = class_weights["entity"]
    model.config.document_class_weights = class_weights["document"]
    model.config.canonical_class_weights = class_weights["canonical"]
    model.config.relation_class_weights = class_weights["relation"]

    accumulation = int(cfg.get("layout_model", {}).get("gradient_accumulation_steps", 4))
    if args.tiny_overfit:
        accumulation = 1
    head_parameters = []
    encoder_parameters = []
    for name, parameter in model.named_parameters():
        (encoder_parameters if name.startswith("layoutlmv2.") else head_parameters).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": 2e-5},
            {"params": head_parameters, "lr": 1e-4 if not args.tiny_overfit else 3e-4},
        ],
        weight_decay=0.01,
    )
    steps_per_epoch = max(1, math.ceil(len(train_dataset) / accumulation))
    planned_optimizer_steps = steps_per_epoch * int(settings["epochs"])
    if settings["max_optimizer_steps"]:
        planned_optimizer_steps = min(planned_optimizer_steps, int(settings["max_optimizer_steps"]))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, round(planned_optimizer_steps * 0.1)),
        num_training_steps=max(1, planned_optimizer_steps),
    )
    use_amp = selected_device.type == "cuda" and bool(
        cfg.get("layout_model", {}).get("mixed_precision", True)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    checkpoint = Path(args.checkpoint) if args.checkpoint else (
        cfgmod.resolve_path(cfg, "ie_checkpoints")
        / "layoutxlm_multitask"
        / args.profile
        / build_id
    )
    checkpoint.mkdir(parents=True, exist_ok=True)
    resume_dir = checkpoint / "resume"

    start_epoch = 0
    resume_batch_index = 0
    optimizer_steps = 0
    micro_steps = 0
    validation_history: list[dict[str, Any]] = []
    rotation_history: list[dict[str, Any]] = []
    best_metric = -math.inf
    best_epoch: int | None = None
    epochs_without_improvement = 0
    if args.resume:
        resume_state_path = Path(args.resume) / "resume_state.pt"
        state = torch.load(resume_state_path, map_location="cpu", weights_only=False)
        if state.get("build_id") != build_id or state.get("profile") != args.profile:
            raise SystemExit("resume checkpoint profile/build does not match the selected manifest")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"])
        resume_batch_index = int(state["next_batch_index"])
        optimizer_steps = int(state["optimizer_steps"])
        micro_steps = int(state["micro_steps"])
        validation_history = list(state.get("validation_history", []))
        rotation_history = list(state.get("rotation_history", []))
        best_metric = float(state.get("best_metric", -math.inf))
        best_epoch = state.get("best_epoch")
        if best_epoch is not None:
            epochs_without_improvement = max(0, len(validation_history) - int(best_epoch))
        _restore_rng_state(state["rng_state"], torch)

    started = time.perf_counter()
    losses: list[float] = []
    task_loss_history: dict[str, list[float]] = defaultdict(list)
    current_epoch = start_epoch
    next_batch_index = resume_batch_index
    stop_requested = False
    stop_reason: str | None = None
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, int(settings["epochs"])):
        current_epoch = epoch
        train_dataset.set_epoch(epoch)
        rotation_history.append(train_dataset.rotation_summary())
        _configure_encoder(model, epoch, int(settings["epochs"]), int(settings["top_layers"]))
        model.train()
        train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed + epoch),
            collate_fn=lambda values: _collate(values, torch),
        )
        for batch_index, batch in enumerate(train_loader):
            if epoch == start_epoch and batch_index < resume_batch_index:
                continue
            next_batch_index = batch_index + 1
            batch = {key: value.to(selected_device) for key, value in batch.items()}
            with torch.amp.autocast("cuda", enabled=use_amp):
                output = model(**batch)
                if output.loss is None:
                    raise RuntimeError("multi-task batch has no valid supervised target")
                loss = output.loss / accumulation
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite multi-task loss")
            scaler.scale(loss).backward()
            micro_steps += 1
            losses.append(float(output.loss.detach().cpu()))
            for name, value in output.task_losses.items():
                task_loss_history[name].append(float(value.detach().cpu()))
            update_now = micro_steps % accumulation == 0 or batch_index + 1 == len(train_loader)
            if update_now:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                if settings["max_optimizer_steps"] and optimizer_steps >= int(settings["max_optimizer_steps"]):
                    stop_requested = True
                    stop_reason = "max_optimizer_steps"
                    break
        if next_batch_index >= len(train_loader):
            current_epoch = epoch + 1
            next_batch_index = 0
        metrics = _evaluate(
            model,
            validation_dataset,
            selected_device,
            torch,
            DataLoader,
        )
        if rotated_validation_dataset is not None:
            rotated_metrics = _evaluate(
                model,
                rotated_validation_dataset,
                selected_device,
                torch,
                DataLoader,
            )
            metrics["rotated_37"] = rotated_metrics
            metrics["selection_composite_score"] = (
                0.7 * float(metrics["composite_score"])
                + 0.3 * float(rotated_metrics["composite_score"])
            )
        else:
            metrics["selection_composite_score"] = float(metrics["composite_score"])
        metrics["epoch"] = epoch + 1
        validation_history.append(metrics)
        selection_metric = float(metrics["selection_composite_score"])
        if selection_metric > best_metric + 1e-9:
            best_metric = selection_metric
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            model.save_pretrained(checkpoint, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint)
        else:
            epochs_without_improvement += 1
            if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
                stop_requested = True
                stop_reason = "early_stopping"
        _save_resume(
            resume_dir,
            model,
            tokenizer,
            optimizer,
            scheduler,
            scaler,
            torch,
            profile=args.profile,
            build_id=build_id,
            epoch=current_epoch,
            next_batch_index=next_batch_index,
            optimizer_steps=optimizer_steps,
            micro_steps=micro_steps,
            validation_history=validation_history,
            rotation_history=rotation_history,
            best_metric=best_metric,
            best_epoch=best_epoch,
        )
        if stop_requested:
            break

    if best_epoch is None:
        raise RuntimeError("training produced no selectable checkpoint")
    selected_model = MultiTaskTextLayoutModel.from_pretrained(checkpoint).to(selected_device).eval()
    validation_metrics = _evaluate(
        selected_model,
        validation_dataset,
        selected_device,
        torch,
        DataLoader,
    )
    if rotated_validation_dataset is not None:
        final_rotated_metrics = _evaluate(
            selected_model,
            rotated_validation_dataset,
            selected_device,
            torch,
            DataLoader,
        )
        validation_metrics["rotated_37"] = final_rotated_metrics
        validation_metrics["selection_composite_score"] = (
            0.7 * float(validation_metrics["composite_score"])
            + 0.3 * float(final_rotated_metrics["composite_score"])
        )
    else:
        validation_metrics["selection_composite_score"] = float(
            validation_metrics["composite_score"]
        )
    reload_result = _reload_check(
        selected_model,
        checkpoint,
        train_dataset,
        selected_device,
        torch,
    )
    training_state = {
        "schema_version": "2.0",
        "profile": args.profile,
        "build_id": build_id,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "public_only": True,
        "gmail_fit_rows": 0,
        "source_checkpoint": checkpoint_id,
        "optimizer_steps": optimizer_steps,
        "micro_steps": micro_steps,
        "best_epoch": best_epoch,
        "best_composite_score": best_metric,
        "checkpoint_selection": {
            "upright_weight": 0.7 if rotated_validation_dataset is not None else 1.0,
            "rotated_37_weight": 0.3 if rotated_validation_dataset is not None else 0.0,
            "split": "dev_select",
        },
        "requested_epochs": int(settings["epochs"]),
        "completed_epochs": len(validation_history),
        "early_stopping_patience": early_stopping_patience,
        "stopped_early": stop_reason == "early_stopping",
        "stop_reason": stop_reason or "completed_requested_epochs",
        "resume_contract": "deterministic epoch/order and optimizer-boundary next_batch_index",
    }
    atomic_write_json(checkpoint / "training_state.json", training_state)
    report = {
        **training_state,
        "status": f"{args.profile}_multitask_trained",
        "architecture": "LayoutXLM-initialized multilingual text-layout encoder with entity, document, canonical-evidence, and real relation heads",
        "visual_backbone": False,
        "license": "CC-BY-NC-SA-4.0",
        "device": str(selected_device),
        "mixed_precision": use_amp,
        "tiny_overfit": bool(args.tiny_overfit),
        "token_sources": sorted(token_sources),
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "train_windows": len(train_dataset),
        "validation_windows": len(validation_dataset),
        "rotated_validation_windows": (
            len(rotated_validation_dataset) if rotated_validation_dataset is not None else 0
        ),
        "training_target_counts": train_dataset.target_counts,
        "dynamic_rotation": {
            "enabled": dynamic_rotation is not None,
            "seed": seed,
            "upright_probability": (
                dynamic_rotation.upright_probability if dynamic_rotation else 1.0
            ),
            "angle_min": dynamic_rotation.angle_min if dynamic_rotation else 0.0,
            "angle_max": dynamic_rotation.angle_max if dynamic_rotation else 0.0,
            "epoch_summaries": rotation_history,
            "geometry_targets_rotated_together": True,
        },
        "class_weights": class_weights,
        "losses": losses,
        "mean_task_losses": {
            name: sum(values) / len(values) for name, values in task_loss_history.items() if values
        },
        "validation": validation_metrics,
        "validation_history": validation_history,
        "checkpoint": str(checkpoint),
        "resume_checkpoint": str(resume_dir),
        "checkpoint_reload_passed": reload_result["passed"],
        "checkpoint_reload_max_difference": reload_result["max_difference"],
        "duration_seconds": time.perf_counter() - started,
        "limitations": [
            "This is a multilingual text-plus-2D-layout model initialized from LayoutXLM-compatible weights; the visual backbone is unavailable on this Windows runtime.",
            "Canonical supervision trains evidence-token field labels; final value generation and abstention are evaluated separately.",
        ],
    }
    report_root = cfgmod.resolve_path(cfg, "reports") / "final_model"
    atomic_write_json(report_root / f"multitask_training_{args.profile}.json", report)
    atomic_write_json(
        cfgmod.resolve_path(cfg, "reports") / "information_extraction" / "layout_model_training.json",
        report,
    )
    print(json.dumps(report, indent=2))
    return 0 if reload_result["passed"] else 1


class TokenizedWindowDataset:
    def __init__(
        self,
        examples: list[dict[str, Any]],
        tokenizer: Any,
        cache_root: Path,
        *,
        max_length: int,
        seed: int,
        torch_module: Any,
        dataset_base: Any,
        dynamic_rotation: Any | None = None,
        fixed_rotation_angle: float | None = None,
    ) -> None:
        del dataset_base
        self.torch = torch_module
        self.entries: list[tuple[Path, int]] = []
        self.preloaded: dict[Path, list[dict[str, Any]]] = {}
        self.examples_by_path: dict[Path, dict[str, Any]] = {}
        self.tokenizer = tokenizer
        self.cache_root = cache_root
        self.max_length = int(max_length)
        self.seed = int(seed)
        self.dynamic_rotation = dynamic_rotation
        if dynamic_rotation is not None and fixed_rotation_angle is not None:
            raise ValueError("dynamic and fixed rotation cannot both be enabled")
        self.fixed_rotation_angle = fixed_rotation_angle
        self.epoch = 0
        counts: Counter[str] = Counter()
        label_counts: dict[str, Counter[int]] = {
            "entity": Counter(),
            "document": Counter(),
            "canonical": Counter(),
            "relation": Counter(),
        }
        cache_root.mkdir(parents=True, exist_ok=True)
        preload = len(examples) <= 2_000
        for example in examples:
            path = cache_root / "upright" / f"{example['example_id']}.pt"
            windows = self._load_windows(example, path, angle=0.0)
            self.examples_by_path[path] = example
            if preload:
                self.preloaded[path] = windows
            for index, window in enumerate(windows):
                self.entries.append((path, index))
                counts["entity_tokens"] += sum(value != -100 for value in window["entity_labels"])
                counts["canonical_tokens"] += sum(value > 0 for value in window["canonical_labels"])
                counts["relation_pairs"] += len(window["relation_pairs"])
                counts["relation_positives"] += sum(
                    pair["label_id"] != 0 for pair in window["relation_pairs"]
                )
                label_counts["entity"].update(
                    value for value in window["entity_labels"] if value != -100
                )
                label_counts["canonical"].update(
                    value for value in window["canonical_labels"] if value != -100
                )
                label_counts["document"].update([int(window["document_label"])])
                label_counts["relation"].update(
                    int(pair["label_id"]) for pair in window["relation_pairs"]
                )
        self.target_counts = dict(counts)
        self.label_counts = label_counts

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        base_path, window_index = self.entries[index]
        example = self.examples_by_path[base_path]
        angle = (
            float(self.fixed_rotation_angle)
            if self.fixed_rotation_angle is not None
            else self.dynamic_rotation.angle_for(str(example["example_id"]), self.epoch)
            if self.dynamic_rotation is not None
            else 0.0
        )
        if abs(angle) < 1e-9:
            path = base_path
        else:
            angle_key = f"{angle:.6f}".replace(".", "p")
            path = self.cache_root / "rotated" / f"{example['example_id']}__a{angle_key}.pt"
        windows = self.preloaded.get(path)
        if windows is None:
            windows = self._load_windows(example, path, angle=angle)
            if len(self.examples_by_path) <= 2_000:
                self.preloaded[path] = windows
        return windows[window_index]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def rotation_summary(self) -> dict[str, Any]:
        angles = [
            self.dynamic_rotation.angle_for(
                str(example["example_id"]), self.epoch
            )
            if self.dynamic_rotation is not None
            else float(self.fixed_rotation_angle)
            if self.fixed_rotation_angle is not None
            else 0.0
            for example in self.examples_by_path.values()
        ]
        upright = sum(abs(angle) < 1e-9 for angle in angles)
        cardinal = sum(
            abs(angle) >= 1e-9
            and min(abs(angle - value) for value in (90.0, 180.0, 270.0, 360.0)) < 1e-6
            for angle in angles
        )
        return {
            "epoch": self.epoch + 1,
            "example_count": len(angles),
            "upright_count": upright,
            "cardinal_count": cardinal,
            "arbitrary_angle_count": len(angles) - upright - cardinal,
            "mean_angle_degrees": sum(angles) / len(angles) if angles else 0.0,
        }

    def _load_windows(
        self,
        example: dict[str, Any],
        path: Path,
        *,
        angle: float,
    ) -> list[dict[str, Any]]:
        if path.is_file():
            return self.torch.load(path, map_location="cpu", weights_only=False)
        if abs(angle) < 1e-9:
            active_example = example
            boxes, _, _, _ = rotated_word_boxes(example, 0.0)
        else:
            active_example, _ = rotate_example_geometry(example, angle)
            boxes = [
                normalize_bbox(
                    token["bbox"],
                    int(active_example["width"]),
                    int(active_example["height"]),
                )
                for token in active_example["tokens"]
            ]
        windows = encode_multitask_windows(
            self.tokenizer,
            active_example,
            boxes=boxes,
            max_length=self.max_length,
            stride=64,
            seed=self.seed,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        self.torch.save(windows, temporary)
        os.replace(temporary, path)
        return windows


def _collate(values: list[dict[str, Any]], torch: Any) -> dict[str, Any]:
    if len(values) != 1:
        raise ValueError("multi-task trainer currently requires batch size 1")
    item = values[0]
    batch = {
        "input_ids": torch.tensor([item["input_ids"]], dtype=torch.long),
        "bbox": torch.tensor([item["bbox"]], dtype=torch.long),
        "attention_mask": torch.tensor([item["attention_mask"]], dtype=torch.long),
        "entity_labels": torch.tensor([item["entity_labels"]], dtype=torch.long),
        "canonical_labels": torch.tensor([item["canonical_labels"]], dtype=torch.long),
        "document_labels": torch.tensor([item["document_label"]], dtype=torch.long),
    }
    if "token_type_ids" in item:
        batch["token_type_ids"] = torch.tensor([item["token_type_ids"]], dtype=torch.long)
    pairs = item["relation_pairs"]
    if pairs:
        batch.update({
            "relation_source_masks": torch.tensor(
                [[pair["source_mask"] for pair in pairs]], dtype=torch.float32
            ),
            "relation_target_masks": torch.tensor(
                [[pair["target_mask"] for pair in pairs]], dtype=torch.float32
            ),
            "relation_geometry": torch.tensor(
                [[pair["geometry"] for pair in pairs]], dtype=torch.float32
            ),
            "relation_source_types": torch.tensor(
                [[pair["source_type_id"] for pair in pairs]], dtype=torch.long
            ),
            "relation_target_types": torch.tensor(
                [[pair["target_type_id"] for pair in pairs]], dtype=torch.long
            ),
            "relation_labels": torch.tensor(
                [[pair["label_id"] for pair in pairs]], dtype=torch.long
            ),
        })
    return batch


def _configure_encoder(model: Any, epoch: int, total_epochs: int, top_layers: int) -> None:
    for parameter in model.layoutlmv2.parameters():
        parameter.requires_grad = False
    if top_layers <= 0 or epoch == 0:
        return
    active_layers = top_layers
    if epoch < total_epochs - 1:
        active_layers = min(4, top_layers)
    for layer in model.layoutlmv2.encoder.layer[-active_layers:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True


def _evaluate(
    model: Any,
    dataset: TokenizedWindowDataset,
    device: Any,
    torch: Any,
    data_loader_class: Any,
    calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    loader = data_loader_class(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda values: _collate(values, torch),
    )
    model.eval()
    losses = []
    entity = Counter()
    canonical = Counter()
    relation = Counter()
    entity_by_class: dict[int, Counter[str]] = defaultdict(Counter)
    canonical_by_class: dict[int, Counter[str]] = defaultdict(Counter)
    relation_by_class: dict[int, Counter[str]] = defaultdict(Counter)
    calibrated_entity = Counter()
    calibrated_canonical = Counter()
    calibrated_relation = Counter()
    calibrated_entity_by_class: dict[int, Counter[str]] = defaultdict(Counter)
    calibrated_canonical_by_class: dict[int, Counter[str]] = defaultdict(Counter)
    calibrated_relation_by_class: dict[int, Counter[str]] = defaultdict(Counter)
    calibrated_document_total = 0
    calibrated_document_retained = 0
    calibrated_document_correct = 0
    document_true: list[int] = []
    document_pred: list[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            if output.loss is not None:
                losses.append(float(output.loss.detach().cpu()))
            entity_predictions = output.entity_logits.argmax(dim=-1)
            canonical_predictions = output.canonical_logits.argmax(dim=-1)
            _update_positive_f1(
                entity,
                entity_predictions,
                batch["entity_labels"],
                positive_threshold=0,
            )
            _update_per_class_f1(
                entity_by_class,
                entity_predictions,
                batch["entity_labels"],
                positive_threshold=0,
            )
            _update_positive_f1(
                canonical,
                canonical_predictions,
                batch["canonical_labels"],
                positive_threshold=0,
            )
            _update_per_class_f1(
                canonical_by_class,
                canonical_predictions,
                batch["canonical_labels"],
                positive_threshold=0,
            )
            document_true.extend(batch["document_labels"].detach().cpu().tolist())
            document_pred.extend(output.document_logits.argmax(dim=-1).detach().cpu().tolist())
            if calibration:
                calibrated_entity_predictions = _calibrated_predictions(
                    output.entity_logits, "entity", calibration, torch,
                )
                calibrated_canonical_predictions = _calibrated_predictions(
                    output.canonical_logits,
                    "canonical",
                    calibration,
                    torch,
                    class_names=CANONICAL_FIELD_LABELS,
                    per_class_thresholds=calibration.get("canonical_field_thresholds"),
                )
                _update_positive_f1(
                    calibrated_entity,
                    calibrated_entity_predictions,
                    batch["entity_labels"],
                    positive_threshold=0,
                )
                _update_per_class_f1(
                    calibrated_entity_by_class,
                    calibrated_entity_predictions,
                    batch["entity_labels"],
                    positive_threshold=0,
                )
                _update_positive_f1(
                    calibrated_canonical,
                    calibrated_canonical_predictions,
                    batch["canonical_labels"],
                    positive_threshold=0,
                )
                _update_per_class_f1(
                    calibrated_canonical_by_class,
                    calibrated_canonical_predictions,
                    batch["canonical_labels"],
                    positive_threshold=0,
                )
                document_probabilities = torch.softmax(
                    output.document_logits
                    / max(0.05, float(calibration["temperatures"]["document"])),
                    dim=-1,
                )
                document_confidences, document_predictions = document_probabilities.max(dim=-1)
                retained = document_confidences.ge(
                    float(calibration["thresholds"]["document"])
                )
                calibrated_document_total += int(retained.numel())
                calibrated_document_retained += int(retained.sum().detach().cpu())
                calibrated_document_correct += int(
                    document_predictions.eq(batch["document_labels"])
                    .logical_and(retained).sum().detach().cpu()
                )
            if output.relation_logits is not None and "relation_labels" in batch:
                relation_predictions = output.relation_logits.argmax(dim=-1)
                _update_positive_f1(
                    relation,
                    relation_predictions,
                    batch["relation_labels"],
                    positive_threshold=0,
                )
                _update_per_class_f1(
                    relation_by_class,
                    relation_predictions,
                    batch["relation_labels"],
                    positive_threshold=0,
                )
                if calibration:
                    calibrated_relation_predictions = _calibrated_predictions(
                        output.relation_logits, "relation", calibration, torch,
                    )
                    _update_positive_f1(
                        calibrated_relation,
                        calibrated_relation_predictions,
                        batch["relation_labels"],
                        positive_threshold=0,
                    )
                    _update_per_class_f1(
                        calibrated_relation_by_class,
                        calibrated_relation_predictions,
                        batch["relation_labels"],
                        positive_threshold=0,
                    )
    entity_f1 = _f1(entity)
    canonical_f1 = _f1(canonical)
    relation_f1 = _f1(relation)
    document_macro_f1 = _macro_f1(document_true, document_pred, len(DOCUMENT_TYPE_LABELS))
    composite = 0.40 * entity_f1 + 0.25 * relation_f1 + 0.25 * canonical_f1 + 0.10 * document_macro_f1
    result = {
        "loss": sum(losses) / len(losses) if losses else None,
        "entity_token_f1": entity_f1,
        "entity_macro_f1": _supported_macro_f1(entity_by_class),
        "entity_per_class": _named_class_metrics(entity_by_class, BIO_LABELS),
        "entity_counts": dict(entity),
        "canonical_evidence_token_f1": canonical_f1,
        "canonical_macro_f1": _supported_macro_f1(canonical_by_class),
        "canonical_per_class": _named_class_metrics(canonical_by_class, CANONICAL_FIELD_LABELS),
        "canonical_counts": dict(canonical),
        "relation_f1": relation_f1,
        "relation_macro_f1": _supported_macro_f1(relation_by_class),
        "relation_per_class": _named_class_metrics(relation_by_class, RELATION_LABELS),
        "relation_counts": dict(relation),
        "document_macro_f1": document_macro_f1,
        "document_accuracy": (
            sum(left == right for left, right in zip(document_true, document_pred))
            / len(document_true)
            if document_true
            else 0.0
        ),
        "composite_score": composite,
        "evaluated_windows": len(dataset),
    }
    if calibration:
        calibrated_entity_f1 = _f1(calibrated_entity)
        calibrated_canonical_f1 = _f1(calibrated_canonical)
        calibrated_relation_f1 = _f1(calibrated_relation)
        result.update({
            "entity_calibrated_f1": calibrated_entity_f1,
            "entity_calibrated_macro_f1": _supported_macro_f1(calibrated_entity_by_class),
            "entity_calibrated_per_class": _named_class_metrics(calibrated_entity_by_class, BIO_LABELS),
            "entity_calibrated_counts": dict(calibrated_entity),
            "canonical_calibrated_f1": calibrated_canonical_f1,
            "canonical_calibrated_macro_f1": _supported_macro_f1(calibrated_canonical_by_class),
            "canonical_calibrated_per_class": _named_class_metrics(calibrated_canonical_by_class, CANONICAL_FIELD_LABELS),
            "canonical_calibrated_counts": dict(calibrated_canonical),
            "relation_calibrated_f1": calibrated_relation_f1,
            "relation_calibrated_macro_f1": _supported_macro_f1(calibrated_relation_by_class),
            "relation_calibrated_per_class": _named_class_metrics(calibrated_relation_by_class, RELATION_LABELS),
            "relation_calibrated_counts": dict(calibrated_relation),
            "document_calibrated_coverage": (
                calibrated_document_retained / calibrated_document_total
                if calibrated_document_total else 0.0
            ),
            "document_calibrated_selective_accuracy": (
                calibrated_document_correct / calibrated_document_retained
                if calibrated_document_retained else None
            ),
            "calibrated_composite_score": (
                0.40 * calibrated_entity_f1
                + 0.25 * calibrated_relation_f1
                + 0.25 * calibrated_canonical_f1
                + 0.10 * document_macro_f1
            ),
        })
    return result


def _calibrated_predictions(
    logits: Any,
    task: str,
    calibration: dict[str, Any],
    torch: Any,
    *,
    class_names: list[str] | tuple[str, ...] | None = None,
    per_class_thresholds: dict[str, float] | None = None,
) -> Any:
    temperature = max(0.05, float(calibration["temperatures"][task]))
    probabilities = torch.softmax(logits / temperature, dim=-1)
    confidences, predictions = probabilities.max(dim=-1)
    retained = apply_abstention(
        confidences.reshape(-1).detach().cpu().tolist(),
        predictions.reshape(-1).detach().cpu().tolist(),
        threshold=float(calibration["thresholds"][task]),
        class_names=class_names,
        per_class_thresholds=per_class_thresholds,
    )
    return torch.tensor(retained, dtype=predictions.dtype).reshape_as(predictions)


def _update_positive_f1(
    counts: Counter[str], predictions: Any, labels: Any, *, positive_threshold: int
) -> None:
    prediction_values = predictions.reshape(-1).detach().cpu().tolist()
    label_values = labels.reshape(-1).detach().cpu().tolist()
    for predicted, actual in zip(prediction_values, label_values):
        if actual == -100:
            continue
        actual_positive = actual > positive_threshold
        predicted_positive = predicted > positive_threshold
        if actual_positive and predicted == actual:
            counts["tp"] += 1
        else:
            if predicted_positive:
                counts["fp"] += 1
            if actual_positive:
                counts["fn"] += 1


def _update_per_class_f1(
    counts: dict[int, Counter[str]],
    predictions: Any,
    labels: Any,
    *,
    positive_threshold: int,
) -> None:
    prediction_values = predictions.reshape(-1).detach().cpu().tolist()
    label_values = labels.reshape(-1).detach().cpu().tolist()
    for predicted, actual in zip(prediction_values, label_values):
        if actual == -100:
            continue
        if actual > positive_threshold:
            if predicted == actual:
                counts[int(actual)]["tp"] += 1
            else:
                counts[int(actual)]["fn"] += 1
        if predicted > positive_threshold and predicted != actual:
            counts[int(predicted)]["fp"] += 1


def _supported_macro_f1(counts: dict[int, Counter[str]]) -> float:
    supported = [values for values in counts.values() if values["tp"] + values["fn"] > 0]
    return sum(_f1(values) for values in supported) / len(supported) if supported else 0.0


def _named_class_metrics(
    counts: dict[int, Counter[str]], labels: list[str] | tuple[str, ...]
) -> dict[str, Any]:
    return {
        str(labels[index] if index < len(labels) else index): {
            **dict(values),
            "f1": _f1(values),
            "support": values["tp"] + values["fn"],
        }
        for index, values in sorted(counts.items())
    }


def _f1(counts: Counter[str]) -> float:
    denominator = 2 * counts["tp"] + counts["fp"] + counts["fn"]
    return 2 * counts["tp"] / denominator if denominator else 0.0


def _macro_f1(truth: list[int], predictions: list[int], class_count: int) -> float:
    scores = []
    for label in range(class_count):
        tp = sum(actual == predicted == label for actual, predicted in zip(truth, predictions))
        fp = sum(actual != label and predicted == label for actual, predicted in zip(truth, predictions))
        fn = sum(actual == label and predicted != label for actual, predicted in zip(truth, predictions))
        denominator = 2 * tp + fp + fn
        if tp + fn:
            scores.append(2 * tp / denominator if denominator else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def _tiny_examples(examples: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    ordered = sorted(
        examples,
        key=lambda example: (
            -int(bool(example.get("relations"))),
            -int(bool(example.get("canonical_fields"))),
            str(example.get("dataset", "")),
            str(example.get("example_id", "")),
        ),
    )
    return ordered[:limit]


def _inverse_sqrt_weights(counts: Counter[int], class_count: int) -> list[float]:
    nonzero = [counts[index] for index in range(class_count) if counts[index] > 0]
    if not nonzero:
        return [1.0] * class_count
    total = sum(nonzero)
    active_count = len(nonzero)
    values = []
    for index in range(class_count):
        count = counts[index]
        if count <= 0:
            values.append(0.0)
            continue
        value = math.sqrt(total / (active_count * count))
        values.append(max(0.25, min(4.0, value)))
    active_mean = sum(value for value in values if value > 0) / active_count
    return [value / active_mean if value > 0 else 0.0 for value in values]


def _save_resume(
    path: Path,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    torch: Any,
    **state: Any,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    payload = {
        **state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng_state": _rng_state(torch),
    }
    temporary = path / "resume_state.pt.tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path / "resume_state.pt")


def _reload_check(
    model: Any,
    checkpoint: Path,
    dataset: TokenizedWindowDataset,
    device: Any,
    torch: Any,
) -> dict[str, Any]:
    from src.information_extraction.layoutxlm_model import MultiTaskTextLayoutModel

    batch = _collate([dataset[0]], torch)
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.no_grad():
        before = model(**batch)
    reloaded = MultiTaskTextLayoutModel.from_pretrained(checkpoint).to(device).eval()
    with torch.no_grad():
        after = reloaded(**batch)
    differences = []
    for name in ("entity_logits", "canonical_logits", "document_logits", "relation_logits"):
        left, right = getattr(before, name), getattr(after, name)
        if left is not None and right is not None:
            differences.append(float((left - right).abs().max().detach().cpu()))
    maximum = max(differences, default=0.0)
    return {"passed": maximum <= 1e-5, "max_difference": maximum}


def _rng_state(torch: Any) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: dict[str, Any], torch: Any) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all(state["cuda"])


def _device(requested: str, torch: Any) -> Any:
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but PyTorch CUDA is unavailable")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def _seed_everything(seed: int, torch: Any) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        pass


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
