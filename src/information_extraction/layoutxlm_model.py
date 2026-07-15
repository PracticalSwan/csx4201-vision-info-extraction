"""Detectron2-free text+layout use of the official LayoutXLM base checkpoint.

The Microsoft checkpoint declares ``model_type: layoutlmv2``. Transformers'
stock LayoutLMv2 model requires Detectron2 solely for its visual backbone,
which has no supported Windows wheel. This model retains and loads the
checkpoint's multilingual word embeddings, 2D spatial embeddings, and all
Transformer encoder layers while omitting only that unavailable visual branch.
It remains a multilingual layout-aware token classifier and does not pretend to
provide image-feature encoding.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.layoutlmv2.modeling_layoutlmv2 import (
    LayoutLMv2Embeddings,
    LayoutLMv2Encoder,
    LayoutLMv2PreTrainedModel,
)


class _TextLayoutBackbone(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.embeddings = LayoutLMv2Embeddings(config)
        self.encoder = LayoutLMv2Encoder(config)

    def forward(
        self,
        input_ids: torch.LongTensor,
        bbox: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_hidden_states: bool = False,
    ):
        input_shape = input_ids.size()
        if bbox is None:
            bbox = torch.zeros((*input_shape, 4), dtype=torch.long, device=input_ids.device)
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, dtype=torch.long, device=input_ids.device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
        if position_ids is None:
            position_ids = self.embeddings.position_ids[:, : input_shape[1]].expand(input_shape)
        word = self.embeddings.word_embeddings(input_ids)
        position = self.embeddings.position_embeddings(position_ids)
        spatial = self.embeddings._calc_spatial_position_embeddings(bbox)
        token_type = self.embeddings.token_type_embeddings(token_type_ids)
        hidden = self.embeddings.LayerNorm(word + position + spatial + token_type)
        hidden = self.embeddings.dropout(hidden)
        extended_mask = attention_mask[:, None, None, :].to(dtype=hidden.dtype)
        extended_mask = (1.0 - extended_mask) * torch.finfo(hidden.dtype).min
        return self.encoder(
            hidden,
            attention_mask=extended_mask,
            head_mask=[None] * len(self.encoder.layer),
            output_attentions=False,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            bbox=bbox,
            position_ids=position_ids,
        )


class LayoutXLMTextLayoutForTokenClassification(LayoutLMv2PreTrainedModel):
    """LayoutXLM multilingual text+2D-layout encoder with a token head."""

    _keys_to_ignore_on_load_unexpected = [
        r"layoutlmv2\.visual.*",
        r"layoutlmv2\.visual_proj.*",
        r"layoutlmv2\.visual_LayerNorm.*",
        r"layoutlmv2\.visual_segment_embedding",
        r"layoutlmv2\.pooler.*",
    ]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.layoutlmv2 = _TextLayoutBackbone(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def get_input_embeddings(self):
        return self.layoutlmv2.embeddings.word_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor,
        bbox: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: bool = False,
        **_: object,
    ) -> TokenClassifierOutput:
        outputs = self.layoutlmv2(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.num_labels), labels.view(-1), ignore_index=-100
            )
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
