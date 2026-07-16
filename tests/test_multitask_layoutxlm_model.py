from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from src.information_extraction.layoutxlm_model import (  # noqa: E402
    MultiTaskTextLayoutModel,
)


def _config():
    config = transformers.LayoutLMv2Config(
        vocab_size=100,
        hidden_size=48,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
        max_2d_position_embeddings=1024,
        coordinate_size=8,
        shape_size=8,
        num_labels=5,
        has_relative_attention_bias=False,
        has_spatial_attention_bias=False,
    )
    config.num_document_labels = 4
    config.num_canonical_labels = 6
    config.num_relation_labels = 3
    config.num_entity_types = 6
    config.relation_geometry_size = 10
    return config


def test_real_multitask_heads_forward_backward_save_and_reload(tmp_path) -> None:
    torch.manual_seed(42)
    model = MultiTaskTextLayoutModel(_config())
    input_ids = torch.randint(0, 100, (2, 16))
    bbox = torch.randint(0, 900, (2, 16, 4))
    bbox[..., 2:] = torch.maximum(bbox[..., :2], bbox[..., 2:])
    attention_mask = torch.ones_like(input_ids)
    entity_labels = torch.randint(0, 5, (2, 16))
    canonical_labels = torch.randint(0, 6, (2, 16))
    document_labels = torch.tensor([1, 2])
    source_masks = torch.zeros(2, 2, 16)
    target_masks = torch.zeros(2, 2, 16)
    source_masks[0, 0, 1:3] = 1
    target_masks[0, 0, 4:6] = 1
    source_masks[0, 1, 7] = 1
    target_masks[0, 1, 9] = 1
    source_masks[1, 0, 2] = 1
    target_masks[1, 0, 5] = 1
    relation_geometry = torch.randn(2, 2, 10)
    source_types = torch.tensor([[1, 2], [3, 0]])
    target_types = torch.tensor([[2, 3], [4, 0]])
    relation_labels = torch.tensor([[1, 0], [2, -100]])

    result = model(
        input_ids=input_ids,
        bbox=bbox,
        attention_mask=attention_mask,
        entity_labels=entity_labels,
        canonical_labels=canonical_labels,
        document_labels=document_labels,
        relation_source_masks=source_masks,
        relation_target_masks=target_masks,
        relation_geometry=relation_geometry,
        relation_source_types=source_types,
        relation_target_types=target_types,
        relation_labels=relation_labels,
    )

    assert result.entity_logits.shape == (2, 16, 5)
    assert result.canonical_logits.shape == (2, 16, 6)
    assert result.document_logits.shape == (2, 4)
    assert result.relation_logits.shape == (2, 2, 3)
    assert set(result.task_losses) == {"entity", "document", "canonical", "relation"}
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert model.relation_head.classifier[0].weight.grad is not None
    assert model.canonical_classifier.weight.grad is not None
    assert model.document_classifier.weight.grad is not None

    model.eval()
    inputs = {
        "input_ids": input_ids,
        "bbox": bbox,
        "attention_mask": attention_mask,
        "relation_source_masks": source_masks,
        "relation_target_masks": target_masks,
        "relation_geometry": relation_geometry,
        "relation_source_types": source_types,
        "relation_target_types": target_types,
    }
    with torch.no_grad():
        before = model(**inputs)
    model.save_pretrained(tmp_path, safe_serialization=True)
    reloaded = MultiTaskTextLayoutModel.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        after = reloaded(**inputs)
    assert torch.allclose(before.entity_logits, after.entity_logits, atol=1e-6)
    assert torch.allclose(before.canonical_logits, after.canonical_logits, atol=1e-6)
    assert torch.allclose(before.document_logits, after.document_logits, atol=1e-6)
    assert torch.allclose(before.relation_logits, after.relation_logits, atol=1e-6)
