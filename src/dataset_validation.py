"""Dataset-specific validation: image/annotation pairing and structure checks.

Annotation contents are never normalized here; we only record and validate the
relationships between images and their annotations on a per-dataset basis so
missing pairs surface as unmatched-file entries.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

UNMATCHED_COLUMNS = [
    "dataset",
    "file_path",
    "file_type",
    "expected_match",
    "match_status",
    "reason",
    "recommended_action",
]


def find_unmatched(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Dispatch to per-dataset matchers and collect unmatched-file rows."""
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_dataset[r["dataset"]].append(r)

    out: list[dict[str, Any]] = []
    out.extend(_match_sroie(by_dataset.get("sroie", [])))
    out.extend(_match_funsd(by_dataset.get("funsd", [])))
    out.extend(_match_fatura(by_dataset.get("fatura", [])))
    out.extend(_match_coru(by_dataset.get("coru", [])))
    out.extend(_flag_unknown(by_dataset.get("unknown", [])))
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unmatched_row(dataset: str, rel: str, file_type: str, expected: str,
                   reason: str, action: str = "review; do not delete") -> dict[str, Any]:
    return {
        "dataset": dataset,
        "file_path": rel,
        "file_type": file_type,
        "expected_match": expected,
        "match_status": "unmatched",
        "reason": reason,
        "recommended_action": action,
    }


def _stems(rows: list[dict[str, Any]], predicate) -> dict[str, list[str]]:
    """Map stem -> list of rel paths for rows satisfying predicate."""
    out: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        if predicate(r):
            stem = Path(r["original_relative_path"]).stem
            out[stem].append(r["current_relative_path"])
    return out


def _report_unpaired(
    dataset: str,
    image_stems: dict[str, list[str]],
    ann_stems: dict[str, list[str]],
    *,
    image_label: str,
    ann_label: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for stem, paths in image_stems.items():
        if stem not in ann_stems:
            for p in paths:
                out.append(_unmatched_row(
                    dataset, p, "image", f"{ann_label} with same stem '{stem}'",
                    f"no matching {ann_label}"))
    for stem, paths in ann_stems.items():
        if stem not in image_stems:
            for p in paths:
                out.append(_unmatched_row(
                    dataset, p, "annotation", f"{image_label} with same stem '{stem}'",
                    f"no matching {image_label}"))
    return out


# ---------------------------------------------------------------------------
# Per-dataset matchers
# ---------------------------------------------------------------------------


def _match_sroie(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    # Group by split (train/test) using the rel path under the dataset root.
    splits: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        rel = r["original_relative_path"]
        # rel looks like SROIE2019/train/img/X00016469670.jpg
        parts = rel.replace("\\", "/").split("/")
        for key in ("train", "test", "val"):
            if key in parts:
                splits[key].append(r)
                break
    for split, srows in splits.items():
        imgs = _stems(srows, lambda r: r["is_image"])
        boxes = _stems(srows, lambda r: _is_txt_in(srows, r, "box"))
        ents = _stems(srows, lambda r: _is_txt_in(srows, r, "entities"))
        out.extend(_report_unpaired(split_tag("sroie", split), imgs, boxes,
                                    image_label="image", ann_label="box(.txt)"))
        out.extend(_report_unpaired(split_tag("sroie", split), imgs, ents,
                                    image_label="image", ann_label="entities(.txt)"))
    return out


def _is_txt_in(_all_rows, r, segment: str) -> bool:
    if not r.get("extension") == ".txt":
        return False
    return segment in r["original_relative_path"].replace("\\", "/").lower()


def _match_funsd(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    splits: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        rel = r["original_relative_path"].replace("\\", "/").lower()
        for key in ("training_data", "testing_data"):
            if key in rel:
                splits[key].append(r)
                break
    for split, srows in splits.items():
        imgs = _stems(srows, lambda r: r["is_image"])
        anns = _stems(srows, lambda r: r["extension"] == ".json")
        out.extend(_report_unpaired(split_tag("funsd", split), imgs, anns,
                                    image_label="image(.png)", ann_label="annotation(.json)"))
    return out


def _match_fatura(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """FATURA pairs images to annotations across multiple annotation formats.

    Annotation filenames encode the source image plus a format suffix, e.g.
    image ``Template10_Instance0.jpg`` has annotations named
    ``Template10_Instance0.json`` (Original_Format),
    ``Template10_Instance0_coco_test.json`` (COCO), and
    ``Template10_Instance0_hugg_test.json`` (HF). An annotation matches an image
    when the image stem equals the annotation stem OR is a prefix of it followed
    by ``_`` (so the suffixed formats are recognized).
    """
    img_stems = {Path(r["original_relative_path"]).stem for r in rows if r["is_image"]}
    ann_rows = [r for r in rows
                if r["is_annotation"] and r["extension"] in (".json", ".xml", ".txt")]

    def base_keys(stem: str) -> list[str]:
        """All left-truncations of a ``_``-split stem, longest first."""
        parts = stem.split("_")
        return ["_".join(parts[:i]) for i in range(len(parts), 0, -1)]

    def matching_image_key(stem: str) -> str | None:
        for key in base_keys(stem):
            if key in img_stems:
                return key
        return None

    matched_image_keys: set[str] = set()
    unmatched_anns: list[dict[str, Any]] = []
    for r in ann_rows:
        stem = Path(r["original_relative_path"]).stem
        key = matching_image_key(stem)
        if key is None:
            unmatched_anns.append(r)
        else:
            matched_image_keys.add(key)

    unmatched_imgs = [r for r in rows if r["is_image"]
                      and Path(r["original_relative_path"]).stem not in matched_image_keys]

    out: list[dict[str, Any]] = []
    for r in unmatched_anns:
        out.append(_unmatched_row(
            "fatura", r["current_relative_path"], "annotation",
            "image with matching base stem", "no matching image"))
    for r in unmatched_imgs:
        out.append(_unmatched_row(
            "fatura", r["current_relative_path"], "image",
            "annotation with matching base stem", "no matching annotation"))
    return out


def _match_coru(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    # Component-level pairing keyed by (component, split) prefix.
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        rel = r["original_relative_path"].replace("\\", "/").lower()
        if rel.startswith("ocr dataset"):
            prefix = "ocr/" + _split_of(rel)
            buckets[prefix].append(r)
        elif rel.startswith("receipt question answering"):
            prefix = "qa/test"
            buckets[prefix].append(r)
        elif rel.startswith("receipt images & key information detection"):
            prefix = "kie/" + _split_of(rel)
            buckets[prefix].append(r)
    for prefix, brows in buckets.items():
        if prefix.startswith("ocr"):
            imgs = _stems(brows, lambda r: r["is_image"])
            txts = _stems(brows, lambda r: r["extension"] == ".txt")
            out.extend(_report_unpaired(split_tag("coru", prefix), imgs, txts,
                                        image_label="line image", ann_label="text(.txt)"))
        elif prefix.startswith("qa"):
            imgs = _stems(brows, lambda r: r["is_image"])
            jsons = _stems(brows, lambda r: r["extension"] == ".json")
            out.extend(_report_unpaired(split_tag("coru", prefix), imgs, jsons,
                                        image_label="image(.jpg)", ann_label="qa(.json)"))
        elif prefix.startswith("kie"):
            imgs = _stems(brows, lambda r: r["is_image"])
            # KIE labels are YOLO-style .txt files under a labels/ subfolder.
            labels = _stems(brows, lambda r: r["extension"] == ".txt"
                            and "/labels/" in "/" + r["original_relative_path"].replace("\\", "/").lower() + "/")
            out.extend(_report_unpaired(split_tag("coru", prefix), imgs, labels,
                                        image_label="image(.jpg)", ann_label="label(.txt)"))
    return out


def _split_of(rel: str) -> str:
    for key in ("train", "val", "test", "dev"):
        if f"/{key}/" in f"/{rel}/":
            return key
    return "unknown"


def _flag_unknown(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(_unmatched_row(
            "unknown", r["current_relative_path"], r["extension"] or "unknown",
            "dataset identification", "dataset could not be identified",
            "resolve manually; do not delete"))
    return out


def split_tag(dataset: str, split: str) -> str:
    return f"{dataset}:{split}"


def write_unmatched_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=UNMATCHED_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
