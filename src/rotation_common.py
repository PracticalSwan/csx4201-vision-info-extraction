"""Shared deterministic helpers for the rotation experiment.

This module contains no fitting logic. It centralizes angle semantics, stable
hashing, atomic artifact writes, path guards, and union-find grouping so every
pipeline stage applies the same safety rules.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

ZONE_LABELS = (1, 2, 3, 4)
ROTATION_DIRECTION = "counterclockwise"
BOUNDARY_ANGLES = (0, 1, 45, 89, 90, 91, 135, 179, 180, 181, 225, 269, 270, 271, 315, 359)


class RotationPipelineError(RuntimeError):
    """Raised when a required pipeline invariant is violated."""


class LeakageError(RotationPipelineError):
    """Raised when public/private or train/evaluation leakage is detected."""


class ArtifactMismatchError(RotationPipelineError):
    """Raised when an artifact does not match the active configuration."""


def normalize_angle(angle: float | int) -> float:
    """Normalize an angle to the half-open interval [0, 360)."""
    value = float(angle)
    if not math.isfinite(value):
        raise ValueError(f"angle must be finite, got {angle!r}")
    normalized = value % 360.0
    if math.isclose(normalized, 360.0, abs_tol=1e-10):
        return 0.0
    if math.isclose(normalized, 0.0, abs_tol=1e-10):
        return 0.0
    return normalized


def get_rotation_zone(angle: float | int) -> int:
    """Return the exact half-open quadrant zone for the angle."""
    normalized = normalize_angle(angle)
    return min(4, int(normalized // 90.0) + 1)


def circular_angular_error(predicted: float, true: float) -> float:
    """Minimum absolute distance between two angles on the unit circle."""
    delta = abs(normalize_angle(predicted) - normalize_angle(true))
    return min(delta, 360.0 - delta)


def signed_correction_angle(estimated_angle: float) -> float:
    """Signed clockwise correction under the positive-counterclockwise convention."""
    return -normalize_angle(estimated_angle)


def rotation_filename(page_id: str, angle: float | int, zone: int | None = None) -> str:
    normalized = normalize_angle(angle)
    if not normalized.is_integer():
        angle_token = f"{normalized:07.3f}".replace(".", "p")
    else:
        angle_token = f"{int(normalized):03d}"
    actual_zone = zone or get_rotation_zone(normalized)
    if actual_zone not in ZONE_LABELS:
        raise ValueError(f"invalid zone: {actual_zone}")
    return f"{page_id}_angle_{angle_token}_zone_{actual_zone}.png"


def stable_digest(value: str, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def stable_id(prefix: str, *parts: object, length: int = 16) -> str:
    payload = "|".join(str(part).replace("\\", "/") for part in parts)
    return f"{prefix}_{stable_digest(payload, length)}"


def canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def configuration_hash(value: Any, length: int = 16) -> str:
    return stable_digest(canonical_json(value), length)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_digest(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return sha256_file(path)


def ensure_not_raw_output(path: Path, project_root: Path) -> None:
    """Refuse to create an artifact inside the configured raw-data tree."""
    raw_root = (project_root / "data" / "raw").resolve()
    resolved = path.resolve()
    if resolved == raw_root or raw_root in resolved.parents:
        raise RotationPipelineError(f"generated output cannot be under raw data: {path}")


def atomic_write_json(path: Path, payload: Any) -> None:
    ensure_parent(path)
    with _temporary_path(path, suffix=".json.tmp") as tmp:
        tmp.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)


def atomic_write_text(path: Path, value: str) -> None:
    ensure_parent(path)
    with _temporary_path(path, suffix=".txt.tmp") as tmp:
        tmp.write_text(value, encoding="utf-8")
        os.replace(tmp, path)


def atomic_write_csv(path: Path, rows: Iterable[Mapping[str, Any]], columns: Sequence[str]) -> None:
    ensure_parent(path)
    with _temporary_path(path, suffix=".csv.tmp") as tmp:
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({column: _csv_value(row.get(column, "")) for column in columns})
        os.replace(tmp, path)


def atomic_save_npz(path: Path, **arrays: Any) -> None:
    ensure_parent(path)
    with _temporary_path(path, suffix=".npz") as tmp:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def disk_free_bytes(path: Path) -> int:
    return int(shutil.disk_usage(path).free)


def disk_total_bytes(path: Path) -> int:
    return int(shutil.disk_usage(path).total)


def human_bytes(value: int | float) -> str:
    number = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(number) < 1024.0:
            return f"{number:.2f} {unit}"
        number /= 1024.0
    return f"{number:.2f} PiB"


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_dataset_filter(values: str | Sequence[str] | None) -> set[str] | None:
    if values is None:
        return None
    if isinstance(values, str):
        parts = values.split(",")
    else:
        parts = list(values)
    cleaned = {part.strip().lower() for part in parts if part and part.strip()}
    return cleaned or None


def deterministic_rank(identifier: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{identifier}".encode("utf-8")).hexdigest()


class UnionFind:
    """Small deterministic disjoint-set structure for leakage groups."""

    def __init__(self, values: Iterable[str] = ()) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}
        for value in values:
            self.add(value)

    def add(self, value: str) -> None:
        if value not in self.parent:
            self.parent[value] = value
            self.rank[value] = 0

    def find(self, value: str) -> str:
        self.add(value)
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left == root_right:
            return
        rank_left, rank_right = self.rank[root_left], self.rank[root_right]
        if rank_left < rank_right or (rank_left == rank_right and root_right < root_left):
            root_left, root_right = root_right, root_left
            rank_left, rank_right = rank_right, rank_left
        self.parent[root_right] = root_left
        if rank_left == rank_right:
            self.rank[root_left] += 1

    def groups(self) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {}
        for value in sorted(self.parent):
            grouped.setdefault(self.find(value), []).append(value)
        return grouped


class _temporary_path:
    def __init__(self, target: Path, suffix: str) -> None:
        self.target = target
        self.suffix = suffix
        self.path: Path | None = None

    def __enter__(self) -> Path:
        ensure_parent(self.target)
        fd, raw = tempfile.mkstemp(prefix=f".{self.target.name}.", suffix=self.suffix, dir=self.target.parent)
        os.close(fd)
        self.path = Path(raw)
        return self.path

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.path is not None and self.path.exists():
            self.path.unlink(missing_ok=True)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict, set)):
        return canonical_json(value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    return value
