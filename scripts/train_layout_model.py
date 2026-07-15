#!/usr/bin/env python3
"""Train, save, reload, and verify the public-only LayoutXLM pre-model."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config as cfgmod  # noqa: E402
from src.information_extraction.layoutxlm_data import (  # noqa: E402
    BIO_LABELS,
    ID_TO_LABEL,
    LABEL_TO_ID,
    encode_layoutxlm_windows,
    load_model_examples,
    rotated_word_boxes,
    to_bio_labels,
)
from src.ocr.environment import configure_external_environment  # noqa: E402
from src.rotation_common import atomic_write_json  # noqa: E402

PROFILE_DEFAULTS = {
    "smoke": {"epochs": 1, "max_steps": 2, "max_length": 128, "freeze_base": True},
    "development": {"epochs": 2, "max_steps": 100, "max_length": 256, "freeze_base": True},
    "final": {"epochs": 5, "max_steps": 0, "max_length": 512, "freeze_base": False},
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--profile", choices=tuple(PROFILE_DEFAULTS), default="smoke")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume", default=None, help="checkpoint directory to resume")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    cfg = cfgmod.load_config(args.config)
    configure_external_environment(cfgmod.resolve_path(cfg, "external_assets"))

    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import LayoutXLMTokenizerFast

    from src.information_extraction.layoutxlm_model import (
        LayoutXLMTextLayoutForTokenClassification,
    )

    settings = dict(PROFILE_DEFAULTS[args.profile])
    if args.max_steps is not None:
        settings["max_steps"] = args.max_steps
    seed = int(cfg.get("layout_model", {}).get("seed", 42))
    _seed_everything(seed, torch)
    selected_device = _device(args.device, torch)
    manifest_path = cfgmod.resolve_path(cfg, "metadata") / "model_dataset_manifest.csv"
    train_examples = load_model_examples(manifest_path, "train")
    validation_examples = load_model_examples(manifest_path, "validation")
    if not train_examples:
        raise SystemExit("no usable public training examples; run prepare_model_dataset.py first")
    if any(example.get("is_private") for example in train_examples + validation_examples):
        raise SystemExit("private model example detected; training refused")
    checkpoint_id = str(cfg.get("layout_model", {}).get("checkpoint", "microsoft/layoutxlm-base"))
    source_checkpoint = args.resume or checkpoint_id
    tokenizer = LayoutXLMTokenizerFast.from_pretrained(
        args.resume or checkpoint_id, cache_dir=str(cfgmod.resolve_path(cfg, "layout_models"))
    )
    model = LayoutXLMTextLayoutForTokenClassification.from_pretrained(
        source_checkpoint,
        cache_dir=str(cfgmod.resolve_path(cfg, "layout_models")),
        num_labels=len(BIO_LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        ignore_mismatched_sizes=not bool(args.resume),
    )
    if settings["freeze_base"]:
        for parameter in model.layoutlmv2.parameters():
            parameter.requires_grad = False
    else:
        layers = model.layoutlmv2.encoder.layer
        for parameter in model.layoutlmv2.parameters():
            parameter.requires_grad = False
        for layer in layers[-2:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
    model.to(selected_device)

    class Examples(Dataset):
        def __init__(self, values: list[dict[str, Any]], training: bool) -> None:
            self.values = values
            self.training = training
            self.epoch = 0
            self.sampled_angles: list[float] = []
            self.windows: list[tuple[int, int]] = []
            for example_index, example in enumerate(values):
                words = [token["text"] for token in example["tokens"]]
                labels = [LABEL_TO_ID[label] for label in to_bio_labels(example["labels"])]
                boxes, _, _, _ = rotated_word_boxes(example, 0.0)
                encoded = encode_layoutxlm_windows(
                    tokenizer,
                    words,
                    boxes,
                    labels,
                    max_length=settings["max_length"],
                    stride=32,
                )
                window_count = _encoding_window_count(encoded)
                self.windows.extend((example_index, window_index) for window_index in range(window_count))

        def __len__(self) -> int:
            return len(self.windows)

        def __getitem__(self, index: int) -> dict[str, Any]:
            example_index, window_index = self.windows[index]
            example = self.values[example_index]
            if self.training:
                material = f"{seed}|{self.epoch}|{example['example_id']}|{window_index}"
                rng = random.Random(material)
                angle = 0.0 if rng.random() < 0.2 else rng.uniform(0.0, 360.0)
            else:
                angle = 0.0
            if self.training:
                self.sampled_angles.append(float(angle))
            boxes, _, _, _ = rotated_word_boxes(example, angle)
            words = [token["text"] for token in example["tokens"]]
            labels = [LABEL_TO_ID[label] for label in to_bio_labels(example["labels"])]
            encoding = encode_layoutxlm_windows(
                tokenizer, words, boxes, labels, max_length=settings["max_length"], stride=32
            )
            item = {}
            for key in ("input_ids", "bbox", "attention_mask", "token_type_ids", "labels"):
                if key in encoding:
                    values = encoding[key]
                    value = values[window_index] if values and isinstance(values[0], list) else values
                    item[key] = torch.tensor(value, dtype=torch.long)
            return item

    train_dataset = Examples(train_examples, training=True)
    validation_dataset = Examples(validation_examples, training=False)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, generator=generator)
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=5e-5, weight_decay=0.01)
    use_amp = selected_device.type == "cuda" and bool(cfg.get("layout_model", {}).get("mixed_precision", True))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    accumulation = int(cfg.get("layout_model", {}).get("gradient_accumulation_steps", 4))
    checkpoint = Path(args.checkpoint) if args.checkpoint else (
        cfgmod.resolve_path(cfg, "ie_checkpoints") / "layoutxlm" / args.profile
    )
    checkpoint.mkdir(parents=True, exist_ok=True)
    best_checkpoint = checkpoint / "best"
    patience_limit = int(cfg.get("layout_model", {}).get("early_stopping_patience", 2))
    started = time.perf_counter()
    losses: list[float] = []
    steps = 0
    optimizer_steps = 0
    best_metric = math.inf
    best_epoch: int | None = None
    epochs_without_improvement = 0
    early_stopped = False
    validation_history: list[dict[str, Any]] = []
    optimizer.zero_grad(set_to_none=True)
    try:
        for epoch in range(int(settings["epochs"])):
            epoch_loss_start = len(losses)
            train_dataset.epoch = epoch
            model.train()
            for batch_index, batch in enumerate(train_loader):
                batch = {key: value.to(selected_device) for key, value in batch.items()}
                with torch.amp.autocast("cuda", enabled=use_amp):
                    output = model(**batch)
                    loss = output.loss / accumulation
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite token-classification loss")
                scaler.scale(loss).backward()
                reached_limit = bool(
                    settings["max_steps"] and steps + 1 >= int(settings["max_steps"])
                )
                if (
                    (batch_index + 1) % accumulation == 0
                    or batch_index + 1 == len(train_loader)
                    or reached_limit
                ):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
                losses.append(float(loss.detach().cpu()) * accumulation)
                steps += 1
                if settings["max_steps"] and steps >= int(settings["max_steps"]):
                    break
            epoch_validation = _evaluate(model, validation_loader, selected_device, torch)
            epoch_losses = losses[epoch_loss_start:]
            selection_metric = (
                float(epoch_validation["loss"])
                if epoch_validation["loss"] is not None
                else (sum(epoch_losses) / len(epoch_losses) if epoch_losses else math.inf)
            )
            validation_history.append({
                "epoch": epoch + 1,
                "selection_metric": selection_metric if math.isfinite(selection_metric) else None,
                **epoch_validation,
            })
            if selection_metric < best_metric - 1e-8:
                best_metric = selection_metric
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                model.save_pretrained(best_checkpoint, safe_serialization=True)
                tokenizer.save_pretrained(best_checkpoint)
            else:
                epochs_without_improvement += 1
                if patience_limit > 0 and epochs_without_improvement >= patience_limit:
                    early_stopped = True
                    break
            if settings["max_steps"] and steps >= int(settings["max_steps"]):
                break
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise SystemExit(
                "LayoutXLM OOM: rerun with --profile smoke or development; batch size is already 1 and OCR is not resident"
            ) from exc
        raise
    if best_epoch is not None:
        model = LayoutXLMTextLayoutForTokenClassification.from_pretrained(best_checkpoint)
        model.to(selected_device)
    validation_metrics = _evaluate(model, validation_loader, selected_device, torch)
    model.save_pretrained(checkpoint, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint)
    atomic_write_json(checkpoint / "training_state.json", {
        "profile": args.profile, "steps": steps, "optimizer_steps": optimizer_steps, "seed": seed,
        "source_checkpoint": checkpoint_id, "public_only": True, "gmail_fit_rows": 0,
        "sampled_angles": train_dataset.sampled_angles,
        "best_epoch": best_epoch, "early_stopped": early_stopped,
    })
    # Save and reload a geometry-aware relation head through the same lifecycle.
    relation_result = _relation_head_smoke(checkpoint, model.config.hidden_size, selected_device, torch)
    model.eval()
    first_batch = next(iter(train_loader))
    first_batch = {key: value.to(selected_device) for key, value in first_batch.items()}
    with torch.no_grad():
        before = model(**first_batch).logits.detach().cpu()
    reloaded = LayoutXLMTextLayoutForTokenClassification.from_pretrained(checkpoint).to(selected_device).eval()
    with torch.no_grad():
        after = reloaded(**first_batch).logits.detach().cpu()
    reload_max_difference = float((before - after).abs().max())
    reload_passed = reload_max_difference <= 1e-5
    report = {
        "schema_version": "1.0", "status": f"{args.profile}_trained",
        "profile": args.profile,
        "architecture": "LayoutXLMTextLayoutForTokenClassification+GeometryAwareRelationHead",
        "visual_backbone": False,
        "layout_features": "multilingual token embeddings plus normalized 2D boxes",
        "source_checkpoint": checkpoint_id, "license": "CC-BY-NC-SA-4.0",
        "device": str(selected_device), "mixed_precision": use_amp,
        "train_examples": len(train_examples), "validation_examples": len(validation_examples),
        "train_windows": len(train_dataset), "validation_windows": len(validation_dataset),
        "gmail_fit_rows": 0, "steps": steps, "optimizer_steps": optimizer_steps, "losses": losses,
        "dynamic_rotation": {
            "sample_count": len(train_dataset.sampled_angles),
            "upright_count": sum(angle == 0.0 for angle in train_dataset.sampled_angles),
            "minimum_angle": min(train_dataset.sampled_angles) if train_dataset.sampled_angles else None,
            "maximum_angle": max(train_dataset.sampled_angles) if train_dataset.sampled_angles else None,
        },
        "final_training_loss": losses[-1] if losses else None,
        "validation": validation_metrics, "checkpoint": str(checkpoint),
        "best_checkpoint": str(best_checkpoint) if best_epoch is not None else None,
        "best_epoch": best_epoch,
        "early_stopped": early_stopped,
        "validation_history": validation_history,
        "checkpoint_reload_passed": reload_passed,
        "checkpoint_reload_max_logit_difference": reload_max_difference,
        "relation_head": relation_result,
        "duration_seconds": time.perf_counter() - started,
        "limitations": (
            ["Smoke training proves pipeline/checkpoint lifecycle only; final model quality is not claimed."]
            if args.profile == "smoke"
            else (["Development training is bounded and is not a final-quality model."] if args.profile == "development" else [])
        ),
    }
    atomic_write_json(
        cfgmod.resolve_path(cfg, "reports") / "information_extraction" / "layout_model_training.json",
        report,
    )
    print(json.dumps(report, indent=2))
    return 0 if reload_passed and relation_result["reload_passed"] else 1


def _encoding_window_count(encoding: dict[str, Any]) -> int:
    input_ids = encoding.get("input_ids", [])
    if not input_ids:
        raise ValueError("LayoutXLM tokenizer returned no windows")
    return len(input_ids) if isinstance(input_ids[0], list) else 1


def _evaluate(model: Any, loader: Any, device: Any, torch: Any) -> dict[str, Any]:
    if len(loader) == 0:
        return {"loss": None, "token_accuracy": None, "evaluated_tokens": 0}
    model.eval()
    losses, correct, total = [], 0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            losses.append(float(output.loss.detach().cpu()))
            predictions = output.logits.argmax(dim=-1)
            mask = batch["labels"] != -100
            correct += int(((predictions == batch["labels"]) & mask).sum())
            total += int(mask.sum())
    return {
        "loss": sum(losses) / len(losses),
        "token_accuracy": correct / total if total else None,
        "evaluated_tokens": total,
    }


def _relation_head_smoke(checkpoint: Path, hidden_size: int, device: Any, torch: Any) -> dict[str, Any]:
    from src.information_extraction.modeling import GeometryAwareRelationHead

    torch.manual_seed(42)
    head = GeometryAwareRelationHead(hidden_size).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
    source = torch.randn(6, hidden_size, device=device)
    target = torch.randn(6, hidden_size, device=device)
    geometry = torch.randn(6, 8, device=device)
    source_types = torch.tensor([1, 1, 2, 3, 4, 1], device=device)
    target_types = torch.tensor([2, 2, 3, 4, 5, 2], device=device)
    labels = torch.tensor([1, 0, 2, 0, 3, 1], device=device)
    logits = head(source, target, geometry, source_types, target_types)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()
    path = checkpoint / "relation_head.pt"
    torch.save(head.state_dict(), path)
    reloaded = GeometryAwareRelationHead(hidden_size).to(device)
    reloaded.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    head.eval(); reloaded.eval()
    with torch.no_grad():
        first = head(source, target, geometry, source_types, target_types)
        second = reloaded(source, target, geometry, source_types, target_types)
    difference = float((first - second).abs().max().detach().cpu())
    return {"loss": float(loss.detach().cpu()), "reload_passed": difference <= 1e-6,
            "reload_max_logit_difference": difference, "path": str(path)}


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


if __name__ == "__main__":
    raise SystemExit(main())
