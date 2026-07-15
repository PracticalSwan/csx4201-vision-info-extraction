from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from src.information_extraction.layoutxlm_model import (  # noqa: E402
    LayoutXLMTextLayoutForTokenClassification,
)


def _config():
    return transformers.LayoutLMv2Config(
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


def test_text_layout_model_forward_backward_save_reload(tmp_path) -> None:
    torch.manual_seed(42)
    model = LayoutXLMTextLayoutForTokenClassification(_config())
    input_ids = torch.randint(0, 100, (2, 16))
    bbox = torch.randint(0, 900, (2, 16, 4))
    bbox[..., 2:] = torch.maximum(bbox[..., :2], bbox[..., 2:])
    labels = torch.randint(0, 5, (2, 16))
    result = model(
        input_ids=input_ids,
        bbox=bbox,
        attention_mask=torch.ones_like(input_ids),
        labels=labels,
    )
    assert result.logits.shape == (2, 16, 5)
    assert torch.isfinite(result.loss)
    result.loss.backward()
    model.eval()
    with torch.no_grad():
        before = model(input_ids=input_ids, bbox=bbox).logits
    model.save_pretrained(tmp_path, safe_serialization=True)
    reloaded = LayoutXLMTextLayoutForTokenClassification.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        after = reloaded(input_ids=input_ids, bbox=bbox).logits
    assert torch.allclose(before, after, atol=1e-6)
