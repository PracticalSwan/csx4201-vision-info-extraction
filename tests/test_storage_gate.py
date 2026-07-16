from __future__ import annotations

import pytest

from src.ocr.environment import evaluate_storage_reserve, require_storage_gate


def test_storage_reserve_accounts_for_anticipated_writes() -> None:
    result = evaluate_storage_reserve(
        c_free_gib=20.0,
        asset_free_gib=40.0,
        minimum_c_free_gib=15.0,
        minimum_asset_free_gib=15.0,
        anticipated_c_gib=1.0,
        anticipated_asset_gib=20.0,
    )

    assert result["passed"] is True
    assert result["projected_c_free_gib"] == 19.0
    assert result["projected_asset_free_gib"] == 20.0


def test_storage_reserve_fails_before_crossing_c_reserve() -> None:
    result = evaluate_storage_reserve(
        c_free_gib=16.0,
        asset_free_gib=100.0,
        minimum_c_free_gib=15.0,
        minimum_asset_free_gib=15.0,
        anticipated_c_gib=2.0,
    )

    assert result["passed"] is False
    assert any("C:" in message for message in result["errors"])


def test_require_storage_gate_raises_with_actionable_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.ocr.environment.storage_gate",
        lambda *args, **kwargs: {
            "passed": False,
            "errors": ["C: projected free space is below reserve"],
        },
    )

    with pytest.raises(RuntimeError, match="projected free space"):
        require_storage_gate("D:/assets", operation="final training")
