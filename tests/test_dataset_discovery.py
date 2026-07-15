"""Tests for dataset discovery and identification confidence."""
from __future__ import annotations

from src import dataset_discovery as dd

from .conftest import make_config


def test_discovery_identifies_all_known_datasets(isolated_project):
    """All four public datasets + gmail are discovered with usable confidence."""
    datasets = dd.discover_datasets(isolated_project["cfg"])
    found = {d.name: d for d in datasets}
    assert {"sroie", "funsd", "fatura", "coru", "gmail"} <= set(found)
    for name, ds in found.items():
        assert ds.confidence in ("high", "medium"), f"{name} confidence too low"
        assert ds.is_resolved


def test_unknown_directory_is_not_assigned(tmp_path):
    """A random unrelated directory must not be flagged as a known dataset."""
    (tmp_path / "data" / "raw" / "public").mkdir(parents=True)
    junk = tmp_path / "data" / "raw" / "public" / "random_photos"
    (junk / "sub").mkdir(parents=True)
    (junk / "sub" / "a.jpg").write_bytes(b"x")
    cfg = make_config(tmp_path)
    datasets = dd.discover_datasets(cfg)
    assert all(d.name != "random_photos" for d in datasets)
    names = {d.name for d in datasets}
    assert "sroie" not in names and "funsd" not in names  # none present


def test_discovery_target_paths_match_config(isolated_project):
    datasets = dd.discover_datasets(isolated_project["cfg"])
    for ds in datasets:
        assert ds.target_path.name == ("gmail" if ds.name == "gmail" else ds.name)


def test_gmail_classification_keywords():
    cases = {
        "Receipt-RCPT-1.pdf": "receipt",
        "Invoice-INV-6493709.pdf": "invoice",
        "fixture_regulation_alpha_9f2c.pdf": "legal_financial",
        "fixture_policy_beta_7d1e.pdf": "legal_financial",
        "fixture_terms_gamma_4a6b.pdf": "legal_financial",
        "AMysteryDoc_zzz.pdf": "unclassified",
    }
    for filename, expected in cases.items():
        cls = dd.classify_gmail_filename(filename)
        assert cls.category == expected, f"{filename} -> {cls.category} (want {expected})"


def test_discovery_is_idempotent_after_organize(isolated_project):
    """Re-running discovery finds datasets at the same canonical target."""
    first = {d.name: d.current_path for d in dd.discover_datasets(isolated_project["cfg"])}
    second = {d.name: d.current_path for d in dd.discover_datasets(isolated_project["cfg"])}
    assert first == second
