"""Atomic, privacy-aware inference artifact writer."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import ImageDraw

from src.inference.document_io import DocumentPage
from src.rotation_common import atomic_write_json, atomic_write_text


def write_document_outputs(
    result: Mapping[str, Any],
    pages: Sequence[DocumentPage],
    output_root: str | Path,
    *,
    force: bool = False,
    save_visualization: bool = False,
) -> Path:
    root = Path(output_root)
    result_path = root / "document_result.json"
    if result_path.exists() and not force:
        raise FileExistsError(f"output already exists; use --force: {result_path}")
    root.mkdir(parents=True, exist_ok=True)
    page_root = root / "pages"
    log_root = root / "logs"
    atomic_write_json(result_path, dict(result))
    for page_result, page in zip(result["pages"], pages, strict=True):
        prefix = f"page_{int(page_result['page_number']):03d}"
        atomic_write_json(page_root / f"{prefix}_ocr.json", page_result["ocr"])
        atomic_write_json(page_root / f"{prefix}_entities.json", page_result["entities"])
        atomic_write_json(page_root / f"{prefix}_relations.json", page_result["key_value_pairs"])
        if save_visualization:
            visualization = page.image.copy()
            draw = ImageDraw.Draw(visualization)
            for word in page_result["ocr"]["words"]:
                points = [tuple(map(float, point)) for point in word["polygon"]]
                draw.line(points + [points[0]], fill="red", width=2)
            target = page_root / f"{prefix}_visualization.png"
            _atomic_save_png(visualization, target)
    log_payload = {
        "document_id": result["document_id"],
        "status": "success",
        "page_count": len(result["pages"]),
        "duration_seconds": result["processing"]["duration_seconds"],
        "private_output": result["processing"]["private_output"],
    }
    atomic_write_text(log_root / "inference.log", json.dumps(log_payload, sort_keys=True) + "\n")
    return result_path


def require_private_output_root(output: str | Path, private_root: str | Path) -> None:
    output_resolved = Path(output).resolve()
    private_resolved = Path(private_root).resolve()
    try:
        output_resolved.relative_to(private_resolved)
    except ValueError as exc:
        raise ValueError(
            f"--private-output requires a destination under the ignored private root: {private_resolved}"
        ) from exc


def _atomic_save_png(image, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, raw = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".png", dir=target.parent)
    os.close(handle)
    temporary = Path(raw)
    try:
        image.save(temporary, format="PNG")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
