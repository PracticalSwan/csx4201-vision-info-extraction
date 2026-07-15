"""Shared pytest fixtures: synthetic dataset builders and a config factory.

Tests NEVER touch the real project data under data/raw. Every fixture builds a
self-contained tree inside pytest's tmp_path.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from src import config as cfgmod


# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------


def make_config(project_root: Path, candidate_roots=("data/raw",)) -> dict:
    """Build a normalized config rooted at tmp_path for isolated tests."""
    raw_cfg = {
        "paths": {
            "project_root": str(project_root),
            "sroie": "data/raw/public/sroie",
            "funsd": "data/raw/public/funsd",
            "fatura": "data/raw/public/fatura",
            "coru": "data/raw/public/coru",
            "gmail_receipts": "data/raw/private/gmail/receipts",
            "gmail_invoices": "data/raw/private/gmail/invoices",
            "gmail_legal_financial": "data/raw/private/gmail/legal_financial_docs",
            "gmail_unclassified": "data/raw/private/gmail/unclassified",
            "metadata": "data/metadata",
        },
        "discovery": {"candidate_roots": list(candidate_roots)},
    }
    return cfgmod._normalize_config(raw_cfg)


# ---------------------------------------------------------------------------
# Synthetic dataset builders
# ---------------------------------------------------------------------------


def _write_png(path: Path, color=(10, 20, 30), size=(8, 8)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _write_jpg(path: Path, color=(10, 20, 30), size=(8, 8)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, "JPEG")


def build_funsd(public_root: Path) -> Path:
    """Minimal FUNSD: training_data + testing_data with image+json pairs."""
    base = public_root / "funsd_forms" / "dataset"
    for split in ("training_data", "testing_data"):
        for stem in ("0000971160", "0000989556"):
            _write_png(base / split / "images" / f"{stem}.png")
            (base / split / "annotations").mkdir(parents=True, exist_ok=True)
            (base / split / "annotations" / f"{stem}.json").write_text(
                json.dumps({"form": [{"box": [0, 0, 1, 1], "text": "x", "label": "other", "words": []}]}),
                encoding="utf-8",
            )
    return public_root / "funsd_forms"


def build_sroie(public_root: Path) -> Path:
    """Minimal SROIE: train/test with box+entities+img triplets."""
    base = public_root / "sroie_receipts" / "SROIE2019"
    for split in ("train", "test"):
        for stem in ("X00016469670", "X00016469671"):
            _write_jpg(base / split / "img" / f"{stem}.jpg")
            (base / split / "box").mkdir(parents=True, exist_ok=True)
            (base / split / "box" / f"{stem}.txt").write_text("1,1,2,1,2,2,1,2,text", encoding="utf-8")
            (base / split / "entities").mkdir(parents=True, exist_ok=True)
            (base / split / "entities" / f"{stem}.txt").write_text(
                json.dumps({"company": "C", "date": "01/01/2019", "address": "A", "total": "1.00"}),
                encoding="utf-8",
            )
    return public_root / "sroie_receipts"


def build_fatura(public_root: Path) -> Path:
    """Minimal FATURA: images + multi-format annotations."""
    base = public_root / "fatura_invoices" / "invoices_dataset_final"
    for stem in ("Template1_Instance0", "Template1_Instance1"):
        _write_jpg(base / "images" / f"{stem}.jpg")
        for fmt_dir, content in (
            ("Annotations/COCO_compatible_format", json.dumps({"images": [], "annotations": []})),
            ("Annotations/layoutlm_HF_format", json.dumps({"tokens": [], "ner_tags": []})),
            ("Annotations/Original_Format", "orig"),
        ):
            p = base / fmt_dir / f"{stem}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
    return public_root / "fatura_invoices"


def build_coru(public_root: Path) -> Path:
    """Minimal CORU: all four components with their native pairings."""
    base = public_root / "coru_receipts"
    # OCR Dataset (jpg+txt)
    for split in ("train", "test"):
        for stem in ("abc_date", "def_line_0"):
            _write_jpg(base / "OCR Dataset" / split / f"{stem}.jpg", size=(6, 6))
            (base / "OCR Dataset" / split / f"{stem}.txt").write_text("text", encoding="utf-8")
    # Receipt Question Answering (jpg+json)
    for stem in ("000186ea", "0002998f"):
        _write_jpg(base / "Receipt Question Answering" / "test" / f"{stem}.jpg", size=(6, 6))
        (base / "Receipt Question Answering" / "test" / f"{stem}.json").write_text(
            json.dumps({"q": "total?", "a": "1.00"}), encoding="utf-8")
    # Receipt Images & Key Information Detection (images+labels)
    for split in ("train", "test"):
        for stem in ("r1", "r2"):
            _write_jpg(base / "Receipt Images & Key Information Detection" / split / "images" / f"{stem}.jpg")
            lbl = base / "Receipt Images & Key Information Detection" / split / "labels" / f"{stem}.txt"
            lbl.parent.mkdir(parents=True, exist_ok=True)
            lbl.write_text("0 0.5 0.5 0.1 0.1", encoding="utf-8")
    # Item Information Extraction (csv)
    ie = base / "Item Information Extraction"
    ie.mkdir(parents=True, exist_ok=True)
    (ie / "train.csv").write_text("receipt_id,total\nabc,1.00\n", encoding="utf-8")
    (ie / "val.csv").write_text("receipt_id,total\ndef,2.00\n", encoding="utf-8")
    (ie / "IE-test.csv").write_text("receipt_id,total\n", encoding="utf-8")
    return base


def build_gmail(private_root: Path) -> Path:
    """Minimal Gmail tree with pre-categorized private PDF placeholders."""
    base = private_root / "gmail_private_test"
    for sub, names in (
        ("receipts", ("Receipt-RCPT-1.pdf",)),
        ("invoices", ("Invoice-INV-1.pdf",)),
        ("legal_financial_docs", ("fixture_regulation_alpha_9f2c.pdf", "fixture_policy_beta_7d1e.pdf")),
    ):
        d = base / sub
        d.mkdir(parents=True, exist_ok=True)
        for n in names:
            (d / n).write_bytes(b"%PDF-1.4\n% test placeholder\n%%EOF\n")
    return base


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_project(tmp_path: Path) -> dict:
    """Build a full synthetic project under tmp_path and return a config + paths."""
    pub = tmp_path / "data" / "raw" / "public"
    pub.mkdir(parents=True)
    priv = tmp_path / "data" / "raw" / "private"
    priv.mkdir(parents=True)
    build_sroie(pub)
    build_funsd(pub)
    build_fatura(pub)
    build_coru(pub)
    build_gmail(priv)
    (tmp_path / "data" / "metadata").mkdir(parents=True, exist_ok=True)
    cfg = make_config(tmp_path)
    return {
        "cfg": cfg,
        "root": tmp_path,
        "public": pub,
        "private": priv,
    }
