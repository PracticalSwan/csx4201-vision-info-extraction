"""Portable runtime readiness checks."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from .runtime import RuntimeSettings, active_python


def _probe(
    command: list[str],
    name: str,
    *,
    cwd: Path | None = None,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    output = (completed.stdout or completed.stderr).strip()
    return {
        "name": name,
        "ok": completed.returncode == 0,
        "detail": output[-1000:],
    }


def report(settings: RuntimeSettings, *, probe: bool = False) -> dict[str, Any]:
    checks = settings.checks()
    if probe and settings.ocr_python.is_file():
        ocr_probe = (
            "from src.ocr.environment import configure_external_environment;"
            f"configure_external_environment({str(settings.asset_root)!r});"
            "import importlib.metadata,paddle,sklearn;"
            "print('paddle',paddle.__version__,'paddleocr',"
            "importlib.metadata.version('paddleocr'),"
            "'sklearn',sklearn.__version__)"
        )
        checks.append(
            _probe(
                [
                    str(settings.ocr_python),
                    "-c",
                    ocr_probe,
                ],
                "OCR imports",
                cwd=settings.home,
                environment=settings.environment(),
            )
        )
    if probe and settings.layout_python.is_file():
        checks.append(
            _probe(
                [
                    str(settings.layout_python),
                    "-c",
                    "import torch,transformers; "
                    "print('torch',torch.__version__,'transformers',transformers.__version__,"
                    "'cuda',torch.cuda.is_available())",
                ],
                "layout imports",
                cwd=settings.home,
                environment=settings.environment(),
            )
        )
    return {
        "status": "ready" if all(item["ok"] for item in checks) else "not_ready",
        "launcher_python": active_python(),
        "device": settings.device,
        "checks": checks,
        "uses_openai_api": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true", help="import OCR/layout frameworks")
    args = parser.parse_args(argv)
    payload = report(RuntimeSettings.load(), probe=args.probe)
    print(json.dumps(payload, indent=2))
    return 0 if payload["status"] == "ready" else 1
