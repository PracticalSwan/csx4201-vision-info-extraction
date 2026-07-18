"""Stable subprocess API around the verified OCR/layout extraction CLI."""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .results import load_result
from .runtime import RuntimeSettings


SUPPORTED_SUFFIXES = {
    ".bmp", ".jpeg", ".jpg", ".pdf", ".png", ".tif", ".tiff", ".webp"
}


class ExtractionError(RuntimeError):
    """Raised when the existing model worker cannot finish an extraction."""


@dataclass(frozen=True)
class ExtractionRun:
    input_path: Path
    output_dir: Path
    result_path: Path
    payload: dict
    command: tuple[str, ...]


def _safe_stem(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._")
    return (stem or "document")[:60]


def default_output_dir(settings: RuntimeSettings, input_path: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = settings.output_root / f"{timestamp}_{_safe_stem(input_path)}"
    candidate = base
    suffix = 2
    while candidate.exists():
        candidate = Path(f"{base}_{suffix}")
        suffix += 1
    return candidate


def build_command(
    settings: RuntimeSettings,
    input_path: Path,
    output_dir: Path,
    *,
    language: str = "auto",
    device: str | None = None,
    max_pages: int | None = None,
    save_visualization: bool = True,
) -> list[str]:
    command = [
        str(settings.ocr_python),
        str(settings.home / "scripts" / "extract_document.py"),
        "--input",
        str(input_path),
        "--output",
        str(output_dir),
        "--config",
        str(settings.config),
        "--model-setup",
        str(settings.model_setup),
        "--model-checkpoint",
        str(settings.layout_checkpoint),
        "--language",
        language,
        "--device",
        device or settings.device,
        "--force",
    ]
    if save_visualization:
        command.append("--save-visualization")
    if max_pages is not None:
        command.extend(["--max-pages", str(max_pages)])
    return command


def run_extraction(
    input_path: str | Path,
    *,
    settings: RuntimeSettings | None = None,
    output_dir: str | Path | None = None,
    language: str = "auto",
    device: str | None = None,
    max_pages: int | None = None,
    save_visualization: bool = True,
    on_log: Callable[[str], None] | None = None,
) -> ExtractionRun:
    runtime = settings or RuntimeSettings.load()
    source = Path(input_path).expanduser().resolve()
    if not source.is_file():
        raise ExtractionError(f"input file not found: {source}")
    if source.suffix.casefold() not in SUPPORTED_SUFFIXES:
        allowed = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise ExtractionError(f"unsupported input type {source.suffix!r}; expected one of: {allowed}")
    if not runtime.ready:
        missing = [item["name"] for item in runtime.checks() if not item["ok"]]
        raise ExtractionError(
            "runtime is not ready; run the setup/doctor first. Missing: "
            + ", ".join(missing)
        )

    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else default_output_dir(runtime, source)
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = build_command(
        runtime,
        source,
        destination,
        language=language,
        device=device,
        max_pages=max_pages,
        save_visualization=save_visualization,
    )
    lines: list[str] = []
    process = subprocess.Popen(
        command,
        cwd=runtime.home,
        env=runtime.environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    for line in process.stdout:
        clean = line.rstrip()
        lines.append(clean)
        if on_log:
            on_log(clean)
    return_code = process.wait()
    if destination.exists():
        (destination / "portable_run.log").write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )
    if return_code != 0:
        detail = "\n".join(lines[-20:]) or "worker exited without output"
        raise ExtractionError(
            f"model extraction failed with exit code {return_code}:\n{detail}"
        )
    result_path = destination / "document_result.json"
    if not result_path.is_file():
        raise ExtractionError(
            "model worker reported success but document_result.json was not created"
        )
    payload = load_result(result_path)
    return ExtractionRun(
        input_path=source,
        output_dir=destination,
        result_path=result_path,
        payload=payload,
        command=tuple(command),
    )


def run_to_json(*args, **kwargs) -> str:
    run = run_extraction(*args, **kwargs)
    return json.dumps(
        {
            "status": "complete",
            "result": str(run.result_path),
            "output_directory": str(run.output_dir),
        },
        indent=2,
    )
