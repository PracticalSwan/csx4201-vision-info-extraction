"""Tests for deterministic, collision-free stable identifiers."""
from __future__ import annotations

from src import stable_ids


def test_file_id_is_deterministic():
    """file_id depends only on (dataset, rel_path), not on walk order."""
    a = stable_ids.file_id("sroie", "SROIE2019/test/img/X00016469670.jpg")
    b = stable_ids.file_id("sroie", "SROIE2019/test/img/X00016469670.jpg")
    assert a == b
    assert a.startswith("sroie_")


def test_file_id_differs_for_different_paths():
    a = stable_ids.file_id("sroie", "a.jpg")
    b = stable_ids.file_id("sroie", "b.jpg")
    assert a != b


def test_file_id_is_path_separator_invariant():
    """Forward vs back slashes must not change the id."""
    fwd = stable_ids.file_id("funsd", "dataset/training_data/images/x.png")
    back = stable_ids.file_id("funsd", "dataset\\training_data\\images\\x.png")
    assert fwd == back


def test_document_id_is_sequential_and_deterministic():
    """Same input set yields the same sequential ids across runs."""
    keys = ["c", "a", "b", "a"]  # duplicate 'a' must collapse
    m1 = stable_ids.assign_document_ids("coru", keys)
    m2 = stable_ids.assign_document_ids("coru", keys)
    assert m1 == m2
    assert m1["a"] == "coru_000000"
    assert m1["b"] == "coru_000001"
    assert m1["c"] == "coru_000002"


def test_document_ids_do_not_collide_in_fixture():
    """Distinct keys get distinct ids in a synthetic fixture."""
    keys = [f"doc{i}" for i in range(50)]
    mapping = stable_ids.assign_document_ids("fatura", keys)
    ids = list(mapping.values())
    assert len(ids) == len(set(ids)), "document ids collided"


def test_detect_collisions_flags_repeats():
    collisions = stable_ids.detect_collisions(["a", "b", "a", "c", "b"])
    assert collisions == {"a": 2, "b": 2}


def test_file_ids_do_not_collide_across_many_paths():
    paths = [f"dir{i}/file{j}.jpg" for i in range(20) for j in range(20)]
    ids = [stable_ids.file_id("coru", p) for p in paths]
    assert len(ids) == len(set(ids)), "file ids collided across 400 paths"
