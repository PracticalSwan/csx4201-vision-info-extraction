"""Resolve a relocatable OCR Model runtime without changing the trained model."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src import config as cfgmod


RUNTIME_FILENAME = "runtime.json"
LOCAL_RUNTIME_FILENAME = "runtime.local.json"


def _resolve(home: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    candidate = path if path.is_absolute() else home / path
    # Do not dereference virtual-environment interpreter symlinks on POSIX.
    # Executing /opt/venvs/ocr/bin/python activates that venv through argv[0],
    # while its resolved target (/usr/local/bin/python) loses the venv packages.
    return Path(os.path.abspath(candidate))


def _python_in(environment: Path) -> Path:
    windows = environment / "Scripts" / "python.exe"
    posix = environment / "bin" / "python"
    if windows.is_file() or os.name == "nt":
        return windows
    return posix


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    # Windows PowerShell 5's `Set-Content -Encoding UTF8` writes a BOM. Accept
    # both BOM and BOM-less UTF-8 so setup remains compatible across hosts.
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"runtime config must contain a JSON object: {path}")
    return dict(payload)


def _usable_local_runtime(home: Path, payload: Mapping[str, Any]) -> bool:
    """Ignore a copied machine-local runtime when its absolute paths vanished."""
    for key in ("ocr_python", "layout_python"):
        value = payload.get(key)
        if value and not _resolve(home, str(value)).is_file():
            return False
    return bool(payload)


@dataclass(frozen=True)
class RuntimeSettings:
    """Concrete paths used by the lightweight launcher."""

    home: Path
    config: Path
    ocr_python: Path
    layout_python: Path
    model_setup: Path
    layout_checkpoint: Path
    asset_root: Path
    output_root: Path
    device: str

    @classmethod
    def load(cls, home: str | Path | None = None) -> "RuntimeSettings":
        resolved_home = Path(
            os.environ.get("OCR_MODEL_HOME")
            or home
            or Path(__file__).resolve().parents[2]
        ).expanduser().resolve()

        portable = _read_json(resolved_home / RUNTIME_FILENAME)
        local = _read_json(resolved_home / LOCAL_RUNTIME_FILENAME)
        runtime = dict(portable)
        if _usable_local_runtime(resolved_home, local):
            runtime.update(local)

        config = _resolve(
            resolved_home,
            os.environ.get("OCR_MODEL_CONFIG")
            or runtime.get("config")
            or "config.yaml",
        )
        cfg = cfgmod.load_config(config)

        def runtime_path(env_name: str, runtime_key: str, fallback: str | Path) -> Path:
            value = os.environ.get(env_name) or runtime.get(runtime_key) or fallback
            return _resolve(resolved_home, str(value))

        paths = cfg.get("paths", {})
        ocr_environment = _resolve(
            resolved_home,
            str(paths.get("ocr_environment", ".runtime/ocr")),
        )
        layout_environment = _resolve(
            resolved_home,
            str(paths.get("layout_environment", ".runtime/layout")),
        )
        configured_layout_python = paths.get("layout_python")
        configured_checkpoint = cfg.get("layout_model", {}).get(
            "inference_checkpoint",
            "assets/checkpoints/layoutxlm_multitask/final",
        )
        configured_asset_root = paths.get("external_assets", "assets")

        return cls(
            home=resolved_home,
            config=config,
            ocr_python=runtime_path(
                "OCR_MODEL_OCR_PYTHON",
                "ocr_python",
                _python_in(ocr_environment),
            ),
            layout_python=runtime_path(
                "OCR_MODEL_LAYOUT_PYTHON",
                "layout_python",
                configured_layout_python or _python_in(layout_environment),
            ),
            model_setup=runtime_path(
                "OCR_MODEL_MODEL_SETUP",
                "model_setup",
                "reports/ocr/model_setup.json",
            ),
            layout_checkpoint=runtime_path(
                "OCR_MODEL_LAYOUT_CHECKPOINT",
                "layout_checkpoint",
                configured_checkpoint,
            ),
            asset_root=runtime_path(
                "OCR_MODEL_ASSET_ROOT",
                "asset_root",
                configured_asset_root,
            ),
            output_root=runtime_path(
                "OCR_MODEL_OUTPUT_ROOT",
                "output_root",
                "outputs",
            ),
            device=str(
                os.environ.get("OCR_MODEL_DEVICE")
                or runtime.get("device")
                or cfg.get("ocr", {}).get("device")
                or "cpu"
            ),
        )

    def environment(self) -> dict[str, str]:
        """Return child-process overrides for the existing pipeline."""
        environment = os.environ.copy()
        environment.update(
            {
                "OCR_MODEL_HOME": str(self.home),
                "OCR_MODEL_ASSET_ROOT": str(self.asset_root),
                "OCR_MODEL_LAYOUT_PYTHON": str(self.layout_python),
                "OCR_MODEL_LAYOUT_MODELS": str(self.asset_root / "cache" / "layoutxlm"),
                "OCR_MODEL_OCR_CACHE": str(self.asset_root / "cache" / "ocr"),
                "PYTHONPATH": str(self.home)
                + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""),
                "PYTHONUTF8": "1",
            }
        )
        return environment

    def checks(self) -> list[dict[str, Any]]:
        """Return fast, side-effect-free readiness checks."""
        model_file = self.layout_checkpoint / "model.safetensors"
        checks = [
            ("home", self.home, self.home.is_dir()),
            ("config", self.config, self.config.is_file()),
            ("OCR Python", self.ocr_python, self.ocr_python.is_file()),
            ("layout Python", self.layout_python, self.layout_python.is_file()),
            ("OCR model manifest", self.model_setup, self.model_setup.is_file()),
            ("layout checkpoint", self.layout_checkpoint, self.layout_checkpoint.is_dir()),
            ("layout weights", model_file, model_file.is_file()),
        ]
        return [
            {"name": name, "ok": ok, "path": str(path)}
            for name, path, ok in checks
        ]

    @property
    def ready(self) -> bool:
        return all(item["ok"] for item in self.checks())


def active_python() -> str:
    """Expose the lightweight interpreter for diagnostics."""
    return sys.executable
