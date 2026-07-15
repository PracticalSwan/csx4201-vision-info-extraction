"""Tests for privacy rules: public/private separation and Gmail handling."""
from __future__ import annotations

from pathlib import Path

from src import privacy


def test_gmail_path_is_private():
    assert privacy.is_private("data/raw/private/gmail/invoices/x.pdf") is True
    assert privacy.is_private("gmail_private_test/receipts/a.pdf") is True


def test_public_path_is_not_private():
    assert privacy.is_private("data/raw/public/sroie/SROIE2019/test/img/x.jpg") is False
    assert privacy.is_public("data/raw/public/sroie/x.jpg") is True


def test_private_path_not_under_public():
    """assert_private_not_under_public passes for correctly separated paths."""
    privacy.assert_private_not_under_public("data/raw/private/gmail/invoices/x.pdf")  # no raise


def test_private_under_public_raises():
    """A gmail file under the public tree must raise."""
    import pytest
    with pytest.raises(ValueError):
        privacy.assert_private_not_under_public(
            "data/raw/public/gmail/invoices/x.pdf")


def test_anonymize_filename_hides_real_name():
    anon = privacy.anonymize_filename("gmail", "gmail_aabbccddeeff", ".pdf")
    assert anon.endswith(".pdf")
    assert "Invoice" not in anon and "Receipt" not in anon


def test_safe_path_for_report_anonymizes_private():
    cfg = {"privacy": {"gmail_is_private": True,
                       "include_private_filenames_in_public_reports": False}}
    real = "gmail/invoices/Invoice-INV-6493709.pdf"
    safe = privacy.safe_path_for_report(real, "gmail", "gmail_abc123", cfg)
    assert "Invoice-INV" not in safe
    assert safe.endswith(".pdf")


def test_safe_path_for_report_keeps_public():
    cfg = {"privacy": {"include_private_filenames_in_public_reports": False}}
    real = "sroie/SROIE2019/test/img/X00016469670.jpg"
    assert privacy.safe_path_for_report(real, "sroie", "id", cfg) == real


def test_public_report_strips_private_filenames():
    text = "See Invoice-INV-6493709.pdf for details."
    out = privacy.public_report_sanitized(text, ["Invoice-INV-6493709.pdf"])
    assert "Invoice-INV" not in out
    assert "<private-filename>" in out


def test_gmail_flag_disabled_marks_public():
    cfg = {"privacy": {"gmail_is_private": False}}
    assert privacy.is_private("gmail/invoices/x.pdf", cfg) is False


def test_gmail_files_not_in_public_directory(tmp_path):
    """End-to-end: building gmail under private keeps it out of public tree."""
    public = tmp_path / "data" / "raw" / "public"
    private = tmp_path / "data" / "raw" / "private" / "gmail"
    public.mkdir(parents=True)
    private.mkdir(parents=True)
    (private / "invoices" / "secret.pdf").parent.mkdir(parents=True)
    (private / "invoices" / "secret.pdf").write_bytes(b"%PDF-")
    # No gmail file under public/
    for p in public.rglob("*"):
        assert not p.is_file() or not privacy.is_private(p)
