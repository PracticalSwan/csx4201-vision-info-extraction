from __future__ import annotations

from src.information_extraction.multitask_data import (
    CANONICAL_FIELD_LABELS,
    RELATION_LABEL_TO_ID,
    build_relation_supervision,
    build_word_supervision,
    encode_multitask_windows,
)


def _example() -> dict:
    return {
        "example_id": "funsd_1__ground_truth",
        "document_type": "form",
        "tokens": [
            {"id": "t0", "text": "Total", "entity_label": "QUESTION"},
            {"id": "t1", "text": "10.00", "entity_label": "ANSWER"},
            {"id": "t2", "text": "Date", "entity_label": "QUESTION"},
        ],
        "labels": ["QUESTION", "ANSWER", "QUESTION"],
        "entity_ids": ["q1", "a1", "q2"],
        "token_loss_mask": [True, True, False],
        "entities": [
            {"id": "q1", "label": "QUESTION", "token_ids": ["t0"], "bbox": [0, 0, 40, 10]},
            {"id": "a1", "label": "ANSWER", "token_ids": ["t1"], "bbox": [50, 0, 90, 10]},
            {"id": "q2", "label": "QUESTION", "token_ids": ["t2"], "bbox": [0, 30, 40, 40]},
        ],
        "relations": [
            {"id": "r1", "type": "QUESTION_ANSWER", "source_id": "q1", "target_id": "a1"}
        ],
        "canonical_fields": {
            "total_amount": {"value": "10.00", "token_ids": ["t1"], "evidence_valid": True}
        },
    }


def test_word_supervision_uses_masks_entity_identity_and_canonical_evidence() -> None:
    supervision = build_word_supervision(_example())
    total_id = CANONICAL_FIELD_LABELS.index("total_amount")

    assert supervision["entity_bio"] == ["B-QUESTION", "B-ANSWER", "B-QUESTION"]
    assert supervision["entity_loss_mask"] == [True, True, False]
    assert supervision["canonical_label_ids"] == [0, total_id, -100]
    assert supervision["document_label_id"] >= 0


def test_relation_supervision_uses_real_positive_and_bounded_negatives() -> None:
    pairs = build_relation_supervision(
        _example(),
        word_ids=[None, 0, 1, 2, None],
        max_negatives_per_positive=3,
        seed=42,
    )

    positives = [pair for pair in pairs if pair["label_id"] != RELATION_LABEL_TO_ID["NO_RELATION"]]
    negatives = [pair for pair in pairs if pair["label_id"] == RELATION_LABEL_TO_ID["NO_RELATION"]]
    assert len(positives) == 1
    assert positives[0]["relation_id"] == "r1"
    assert positives[0]["label_id"] == RELATION_LABEL_TO_ID["QUESTION_ANSWER"]
    assert positives[0]["source_mask"] == [0.0, 1.0, 0.0, 0.0, 0.0]
    assert positives[0]["target_mask"] == [0.0, 0.0, 1.0, 0.0, 0.0]
    assert len(positives[0]["geometry"]) == 10
    assert len(negatives) <= 3
    assert all(pair["relation_id"] is None for pair in negatives)


class _FakeEncoding(dict):
    def word_ids(self, batch_index: int) -> list[int | None]:
        assert batch_index == 0
        return [None, 0, 0, 1, 2, None, None]


class _FakeTokenizer:
    def __call__(self, words, **kwargs):
        assert words == ["Total", "10.00", "Date"]
        assert kwargs["return_overflowing_tokens"] is True
        return _FakeEncoding({
            "input_ids": [[0, 10, 11, 12, 13, 2, 0]],
            "bbox": [[[0, 0, 0, 0]] * 7],
            "attention_mask": [[1, 1, 1, 1, 1, 1, 0]],
            "token_type_ids": [[0] * 7],
        })


def test_window_encoding_masks_repeat_subwords_and_carries_real_relation_pairs() -> None:
    windows = encode_multitask_windows(
        _FakeTokenizer(),
        _example(),
        boxes=[[0, 0, 10, 10], [20, 0, 30, 10], [0, 20, 10, 30]],
        max_length=7,
        stride=1,
        seed=42,
    )
    window = windows[0]

    assert window["entity_labels"][1] >= 0
    assert window["entity_labels"][2] == -100
    assert window["entity_labels"][4] == -100
    assert window["canonical_labels"][3] == CANONICAL_FIELD_LABELS.index("total_amount")
    assert window["document_label"] >= 0
    assert any(
        pair["label_id"] == RELATION_LABEL_TO_ID["QUESTION_ANSWER"]
        for pair in window["relation_pairs"]
    )
