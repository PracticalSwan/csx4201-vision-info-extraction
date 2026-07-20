from pathlib import Path

from scripts.build_portable_release import copy_application


def test_portable_builder_includes_license_and_contribution_policy(
    tmp_path: Path,
) -> None:
    target = tmp_path / "OCR_Model"

    copy_application(target)

    license_text = (target / "LICENSE").read_text(encoding="utf-8")
    contributing_text = (target / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert "MIT License" in license_text
    assert "Copyright (c) 2026 Sithu Win San" in license_text
    assert "solo academic project" in contributing_text
    assert "pull requests" in contributing_text
    assert "welcome" in contributing_text
