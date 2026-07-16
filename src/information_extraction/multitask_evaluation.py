"""Validation helpers for public multi-task checkpoint evaluation."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def validate_evaluation_binding(
    training_state: Mapping[str, Any],
    *,
    expected_profile: str,
    evaluation_build_id: str,
    allow_cross_build: bool,
) -> None:
    """Refuse accidental cross-profile or cross-build evaluation."""
    checkpoint_profile = str(training_state.get("profile", ""))
    checkpoint_build = str(training_state.get("build_id", ""))
    if checkpoint_profile != expected_profile:
        raise ValueError(
            "checkpoint profile mismatch: "
            f"expected {expected_profile!r}, found {checkpoint_profile!r}"
        )
    if checkpoint_build != evaluation_build_id and not allow_cross_build:
        raise ValueError(
            "checkpoint build mismatch; pass --allow-cross-build only for an "
            "explicit same-profile comparison: "
            f"checkpoint={checkpoint_build!r}, evaluation={evaluation_build_id!r}"
        )
