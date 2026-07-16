"""Configuration loading and path resolution.

All paths in config.yaml are resolved through pathlib so the toolkit works on
Windows and POSIX hosts. No absolute paths are hard-coded in the modules; any
absolute dataset paths live only in config.yaml.
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Mapping

import yaml

DEFAULT_CONFIG_NAME = "config.yaml"

# Canonical dataset keys that have an entry under config["paths"].
DATASET_KEYS = ("sroie", "funsd", "fatura", "coru")
GMAIL_KEYS = (
    "gmail_receipts",
    "gmail_invoices",
    "gmail_legal_financial",
    "gmail_unclassified",
)


class ConfigError(Exception):
    """Raised when the configuration is missing required structure."""


def load_config(config_path: str | Path = DEFAULT_CONFIG_NAME) -> dict[str, Any]:
    """Load a YAML config file into a plain dict.

    Raises ConfigError if the file cannot be read or parsed.
    """
    path = Path(config_path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:  # pragma: no cover - exercised via tests
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ConfigError(f"Config root must be a mapping in {path}")
    return _normalize_config(dict(data))


def _normalize_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Ensure expected top-level sections exist so downstream code can rely on them."""
    cfg.setdefault("paths", {})
    cfg["paths"].setdefault("project_root", ".")
    cfg.setdefault("discovery", {})
    cfg["discovery"].setdefault("candidate_roots", ["data/raw", "vision_info_extraction_data"])
    cfg.setdefault("organization", {})
    cfg["organization"].setdefault("default_mode", "reference")
    cfg["organization"].setdefault("preserve_internal_structure", True)
    cfg["organization"].setdefault("overwrite_existing", False)
    cfg["organization"].setdefault("verify_after_move", True)
    cfg.setdefault("audit", {})
    cfg["audit"].setdefault("calculate_sha256", True)
    cfg["audit"].setdefault("inspect_images", True)
    cfg["audit"].setdefault("inspect_pdfs", True)
    cfg["audit"].setdefault("inspect_annotations", True)
    cfg["audit"].setdefault("continue_on_error", True)
    cfg["audit"].setdefault("sha256_chunk_size", 1 << 20)
    cfg.setdefault("duplicates", {})
    cfg["duplicates"].setdefault("exact_enabled", True)
    cfg["duplicates"].setdefault("perceptual_enabled", True)
    cfg["duplicates"].setdefault("perceptual_threshold", 5)
    cfg["duplicates"].setdefault("max_images_for_full_near_duplicate_scan", 10000)
    cfg.setdefault("privacy", {})
    cfg["privacy"].setdefault("gmail_is_private", True)
    cfg["privacy"].setdefault("include_private_filenames_in_public_reports", False)
    cfg["privacy"].setdefault("include_private_contents_in_reports", False)
    cfg.setdefault("runtime", {})
    cfg["runtime"].setdefault("workers", 2)
    cfg["runtime"].setdefault("log_level", "INFO")
    cfg.setdefault("information_extraction", {})
    cfg["information_extraction"].setdefault("schema_version", "1.0")
    cfg["information_extraction"].setdefault(
        "output_schema", "schemas/inference_output.schema.json"
    )
    cfg["information_extraction"].setdefault(
        "entity_labels", ["HEADER", "KEY", "VALUE", "QUESTION", "ANSWER", "TABLE_CELL", "OTHER"]
    )
    cfg["information_extraction"].setdefault("public_only_training", True)
    cfg.setdefault("augmentation", {})
    cfg["augmentation"].setdefault("upright_probability", 0.2)
    cfg["augmentation"].setdefault("angle_min", 0.0)
    cfg["augmentation"].setdefault("angle_max", 360.0)
    cfg.setdefault("ocr", {})
    cfg["ocr"].setdefault("device", "cpu")
    cfg["ocr"].setdefault("orientation_candidates", [0, 90, 180, 270])
    cfg["ocr"].setdefault("preprocessing_version", "1.0")
    cfg.setdefault("layout_model", {})
    cfg["layout_model"].setdefault("checkpoint", "microsoft/layoutxlm-base")
    cfg["layout_model"].setdefault("max_length", 512)
    cfg.setdefault("kmeans_display", {})
    cfg["kmeans_display"].setdefault("enabled", True)
    cfg["kmeans_display"].setdefault("purpose", "display_only")
    cfg["kmeans_display"].setdefault("experimental_exact_angle_enabled", False)
    return cfg


def project_root(cfg: Mapping[str, Any]) -> Path:
    """Return the resolved project root Path."""
    return Path(cfg["paths"]["project_root"]).resolve()


def resolve_path(cfg: Mapping[str, Any], key: str) -> Path:
    """Resolve a config path key relative to the project root.

    Accepts either a key under paths (e.g. "sroie") or any relative/absolute
    path string. Absolute paths are returned unchanged.
    """
    paths = cfg.get("paths", {})
    raw = paths.get(key, key)
    p = Path(raw)
    if p.is_absolute():
        return p
    return project_root(cfg) / p


def setup_logging(cfg: Mapping[str, Any], level: str | None = None) -> logging.Logger:
    """Configure a console logger; idempotent across repeated calls."""
    level_name = (level or cfg.get("runtime", {}).get("log_level", "INFO")).upper()
    logger = logging.getLogger("vix")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level_name, logging.INFO))
    return logger


def clone(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-copy the config (used by organize/audit to layer overrides)."""
    return copy.deepcopy(dict(cfg))
