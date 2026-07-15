"""Shared helpers for the rotation-pipeline command-line entry points."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config as cfgmod  # noqa: E402


def configure_utf8() -> None:
    """Keep Windows console output readable without making output failures fatal."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass


def parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config.yaml"),
        help="configuration file (default: project config.yaml)",
    )
    ap.add_argument("--log-level", default=None, help="override the configured log level")
    return ap


def load(args: argparse.Namespace) -> dict[str, Any]:
    cfg = cfgmod.load_config(args.config)
    cfgmod.setup_logging(cfg, getattr(args, "log_level", None))
    return cfg


def add_filter_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--limit", type=nonnegative_int, default=0, help="debug row/page cap (0 = no cap)")
    ap.add_argument("--datasets", default=None, help="comma-separated dataset filter")
    ap.add_argument("--splits", default=None, help="comma-separated project-split filter")
    ap.add_argument("--workers", type=positive_int, default=None, help="worker count override")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be 0 or greater")
    return parsed


def print_result(label: str, result: Mapping[str, Any] | Any) -> None:
    """Print a useful bounded summary without dumping manifests or private rows."""
    print(f"{label} complete")
    print(json.dumps(_compact(result), indent=2, sort_keys=True, default=_json_default))


def _compact(value: Any) -> Any:
    if isinstance(value, list):
        if len(value) > 50:
            return {"omitted_count": len(value)}
        return [_compact(item) for item in value]
    if not isinstance(value, Mapping):
        return value
    output: dict[str, Any] = {}
    omitted_counts: dict[str, int] = {}
    for key, item in value.items():
        if key in {"pages", "rows"} and isinstance(item, list):
            omitted_counts[key] = len(item)
        elif key == "errors" and isinstance(item, list):
            output["error_count"] = len(item)
        elif key == "fit_rotation_ids" and isinstance(item, list):
            output["fit_rotation_id_count"] = len(item)
        else:
            output[str(key)] = _compact(item)
    if omitted_counts:
        output["omitted_record_counts"] = omitted_counts
    return output


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


configure_utf8()
