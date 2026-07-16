"""Persistent subprocess client isolating PyTorch from Paddle's CUDA DLLs."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any, Mapping


class SubprocessLayoutEntityExtractor:
    """Keep the LayoutXLM model in a dedicated PyTorch process on Windows."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        python_executable: str | Path | None = None,
        device: str = "cpu",
        cache_dir: str | Path | None = None,
        max_length: int = 512,
        calibration_path: str | Path | None = None,
        confidence_threshold: float | None = None,
        timeout_seconds: float = 300.0,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.is_dir():
            raise FileNotFoundError(f"layout model checkpoint is missing: {self.checkpoint}")
        self.device = device
        self.python_executable = Path(python_executable or sys.executable)
        if not self.python_executable.is_file():
            raise FileNotFoundError(f"layout Python executable is missing: {self.python_executable}")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_length = int(max_length)
        self.calibration_path = Path(calibration_path) if calibration_path else None
        self.confidence_threshold = confidence_threshold
        self.timeout_seconds = float(timeout_seconds)
        self.process: subprocess.Popen[str] | None = None
        self._reader = ThreadPoolExecutor(max_workers=1, thread_name_prefix="layout-worker-read")

    def extract(
        self,
        ocr_result: Mapping[str, Any],
        *,
        page_number: int,
        width: int,
        height: int,
    ) -> dict[str, Any]:
        process = self._ensure_process()
        assert process.stdin is not None and process.stdout is not None
        request = {
            "action": "extract",
            "ocr_result": dict(ocr_result),
            "page_number": int(page_number),
            "width": int(width),
            "height": int(height),
        }
        try:
            process.stdin.write(json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError(self._worker_error("layout worker pipe closed")) from exc
        future = self._reader.submit(process.stdout.readline)
        try:
            line = future.result(timeout=self.timeout_seconds)
        except FutureTimeoutError as exc:
            self.close(kill=True)
            raise TimeoutError(f"layout worker exceeded {self.timeout_seconds:.0f} seconds") from exc
        if not line:
            raise RuntimeError(self._worker_error("layout worker exited without a response"))
        response = json.loads(line)
        if response.get("status") != "ok":
            raise RuntimeError(str(response.get("error") or "layout worker failed"))
        return {
            "entities": list(response.get("entities") or []),
            "relations": list(response.get("relations") or []),
            "canonical_fields": dict(response.get("canonical_fields") or {}),
            "tables": list(response.get("tables") or []),
            "document_type": dict(
                response.get("document_type")
                or {"label": "unknown", "confidence": None}
            ),
            "warnings": list(response.get("warnings") or []),
        }

    def close(self, *, kill: bool = False) -> None:
        process, self.process = self.process, None
        if process is None:
            return
        if process.poll() is None and not kill and process.stdin is not None:
            try:
                process.stdin.write('{"action":"shutdown"}\n')
                process.stdin.flush()
                process.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                kill = True
        if kill and process.poll() is None:
            process.kill()
            process.wait(timeout=10)

    def _ensure_process(self) -> subprocess.Popen[str]:
        if self.process is not None and self.process.poll() is None:
            return self.process
        script = Path(__file__).resolve().parents[2] / "scripts" / "layout_entity_worker.py"
        command = [
            str(self.python_executable),
            str(script),
            "--checkpoint",
            str(self.checkpoint),
            "--device",
            self.device,
            "--max-length",
            str(self.max_length),
        ]
        if self.cache_dir:
            command.extend(["--cache-dir", str(self.cache_dir)])
        if self.calibration_path:
            command.extend(["--calibration", str(self.calibration_path)])
        if self.confidence_threshold is not None:
            command.extend(["--confidence-threshold", str(self.confidence_threshold)])
        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=environment,
            creationflags=creationflags,
        )
        return self.process

    def _worker_error(self, prefix: str) -> str:
        process = self.process
        details = ""
        if process is not None and process.poll() is not None and process.stderr is not None:
            details = process.stderr.read()[-4000:]
        return f"{prefix}{': ' + details.strip() if details.strip() else ''}"

    def __enter__(self):
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown timing
        try:
            self.close(kill=True)
        except Exception:
            pass
