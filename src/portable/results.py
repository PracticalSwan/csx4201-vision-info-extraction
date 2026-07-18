"""Read and present local extraction outputs."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping


def load_result(path: str | Path) -> dict[str, Any]:
    result_path = Path(path)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or "pages" not in payload or "fields" not in payload:
        raise ValueError(f"not a document extraction result: {result_path}")
    return dict(payload)


def field_rows(payload: Mapping[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for name, evidence in sorted(dict(payload.get("fields") or {}).items()):
        item = dict(evidence or {})
        if item.get("value") in {None, ""}:
            continue
        confidence = item.get("confidence")
        if isinstance(confidence, (int, float)):
            confidence = round(float(confidence), 4)
        rows.append(
            [
                name,
                item.get("value"),
                confidence,
                item.get("method") or item.get("extraction_source"),
                item.get("page_number"),
                item.get("validation_status"),
            ]
        )
    return rows


def ocr_text(payload: Mapping[str, Any]) -> str:
    pages = list(payload.get("pages") or [])
    chunks = []
    for page in pages:
        number = page.get("page_number", len(chunks) + 1)
        chunks.append(f"--- Page {number} ---\n{str(page.get('full_text') or '').strip()}")
    return "\n\n".join(chunks).strip()


def visualization_files(output_dir: str | Path) -> list[str]:
    return [
        str(path)
        for path in sorted((Path(output_dir) / "pages").glob("page_*_visualization.png"))
        if path.is_file()
    ]


def create_result_archive(output_dir: str | Path) -> Path:
    directory = Path(output_dir).resolve()
    archive = directory.with_suffix(".zip")
    if archive.exists():
        archive.unlink()
    return Path(shutil.make_archive(str(directory), "zip", root_dir=directory))
