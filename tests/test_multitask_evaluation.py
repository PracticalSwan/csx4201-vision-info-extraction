from __future__ import annotations

import pytest

from src.information_extraction.multitask_evaluation import (
    validate_evaluation_binding,
)


def test_evaluation_binding_requires_explicit_cross_build_comparison() -> None:
    state = {"profile": "development", "build_id": "old-build"}

    validate_evaluation_binding(
        state,
        expected_profile="development",
        evaluation_build_id="old-build",
        allow_cross_build=False,
    )
    with pytest.raises(ValueError, match="build"):
        validate_evaluation_binding(
            state,
            expected_profile="development",
            evaluation_build_id="new-build",
            allow_cross_build=False,
        )
    validate_evaluation_binding(
        state,
        expected_profile="development",
        evaluation_build_id="new-build",
        allow_cross_build=True,
    )


def test_evaluation_binding_never_allows_profile_mismatch() -> None:
    with pytest.raises(ValueError, match="profile"):
        validate_evaluation_binding(
            {"profile": "smoke", "build_id": "build"},
            expected_profile="development",
            evaluation_build_id="build",
            allow_cross_build=True,
        )
