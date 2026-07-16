"""General/Thai route decisions based on metadata and OCR evidence."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SUPPORTED_LANGUAGE_MODES = {"auto", "general", "thai", "en", "tr", "th"}
GENERAL_CONFIDENCE_RETRY_THRESHOLD = 0.75


def normalize_language_mode(mode: str | None) -> str:
    normalized = (mode or "auto").strip().lower()
    if normalized not in SUPPORTED_LANGUAGE_MODES:
        raise ValueError(f"unsupported language mode: {mode}")
    return normalized


def route_for_mode(mode: str) -> str | None:
    normalized = normalize_language_mode(mode)
    if normalized in {"general", "en", "tr"}:
        return "general"
    if normalized in {"thai", "th"}:
        return "thai"
    return None


def contains_thai(text: str) -> bool:
    return any("\u0e00" <= character <= "\u0e7f" for character in text)


def should_try_thai(
    general_result: Mapping[str, Any],
    general_score: Mapping[str, float],
    *,
    language_hint: str | None = None,
    metadata_language: str | None = None,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    hints = {str(language_hint or "").lower(), str(metadata_language or "").lower()}
    if hints & {"th", "thai", "th-th"}:
        reasons.append("metadata_or_user_hint_is_thai")
    if contains_thai(str(general_result.get("full_text", ""))):
        reasons.append("general_output_contains_thai_script")
    if general_score.get("mean_confidence", 0.0) < GENERAL_CONFIDENCE_RETRY_THRESHOLD:
        reasons.append("general_confidence_is_low")
    if general_score.get("word_count", 0.0) < 2:
        reasons.append("general_output_is_empty_or_weak")
    if general_score.get("valid_character_ratio", 0.0) < 0.8:
        reasons.append("general_output_has_low_valid_character_ratio")
    return bool(reasons), reasons


def select_route_result(
    general: tuple[dict[str, Any], Mapping[str, float]] | None,
    thai: tuple[dict[str, Any], Mapping[str, float]] | None,
    *,
    preferred_route: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = [candidate for candidate in (general, thai) if candidate is not None]
    if not candidates:
        raise ValueError("at least one OCR route result is required")
    selected, selected_score = max(
        candidates,
        key=lambda pair: (
            float(pair[1].get("total", 0.0))
            + (0.08 if pair[0].get("language_route") == preferred_route else 0.0),
            pair[0].get("language_route") == "general",
        ),
    )
    decision = {
        "selected_route": selected["language_route"],
        "reason": (
            "metadata_preference_plus_route_calibrated_score"
            if preferred_route else "highest_route_calibrated_candidate_score"
        ),
        "preferred_route": preferred_route,
        "scores": {pair[0]["language_route"]: dict(pair[1]) for pair in candidates},
    }
    return selected, decision
