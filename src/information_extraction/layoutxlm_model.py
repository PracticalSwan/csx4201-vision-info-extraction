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

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.layoutlmv2.modeling_layoutlmv2 import (
    LayoutLMv2Embeddings,
    LayoutLMv2Encoder,
    LayoutLMv2PreTrainedModel,
)
from transformers.utils import ModelOutput

from src.information_extraction.modeling import GeometryAwareRelationHead


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


@dataclass
class MultiTaskTextLayoutOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    entity_logits: Optional[torch.Tensor] = None
    document_logits: Optional[torch.Tensor] = None
    canonical_logits: Optional[torch.Tensor] = None
    relation_logits: Optional[torch.Tensor] = None
    task_losses: Optional[dict[str, torch.Tensor]] = None
    hidden_states: Optional[tuple[torch.Tensor, ...]] = None


class MultiTaskTextLayoutModel(LayoutLMv2PreTrainedModel):
    """Multilingual text+2D-layout encoder with four genuinely trained heads."""

    _keys_to_ignore_on_load_unexpected = LayoutXLMTextLayoutForTokenClassification._keys_to_ignore_on_load_unexpected

    def __init__(self, config) -> None:
        super().__init__(config)
        self.num_entity_labels = int(config.num_labels)
        self.num_document_labels = int(getattr(config, "num_document_labels", 4))
        self.num_canonical_labels = int(getattr(config, "num_canonical_labels", 2))
        self.num_relation_labels = int(getattr(config, "num_relation_labels", 2))
        self.num_entity_types = int(getattr(config, "num_entity_types", 10))
        self.relation_geometry_size = int(getattr(config, "relation_geometry_size", 10))
        self.layoutlmv2 = _TextLayoutBackbone(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.entity_classifier = nn.Linear(config.hidden_size, self.num_entity_labels)
        self.document_classifier = nn.Linear(config.hidden_size, self.num_document_labels)
        self.canonical_classifier = nn.Linear(config.hidden_size, self.num_canonical_labels)
        self.relation_head = GeometryAwareRelationHead(
            config.hidden_size,
            geometry_size=self.relation_geometry_size,
            entity_type_count=self.num_entity_types,
            relation_type_count=self.num_relation_labels,
        )
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
        entity_labels: Optional[torch.LongTensor] = None,
        document_labels: Optional[torch.LongTensor] = None,
        canonical_labels: Optional[torch.LongTensor] = None,
        relation_source_masks: Optional[torch.Tensor] = None,
        relation_target_masks: Optional[torch.Tensor] = None,
        relation_geometry: Optional[torch.Tensor] = None,
        relation_source_types: Optional[torch.LongTensor] = None,
        relation_target_types: Optional[torch.LongTensor] = None,
        relation_labels: Optional[torch.LongTensor] = None,
        output_hidden_states: bool = False,
        **_: object,
    ) -> MultiTaskTextLayoutOutput:
        outputs = self.layoutlmv2(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
        )
        sequence = self.dropout(outputs.last_hidden_state)
        entity_logits = self.entity_classifier(sequence)
        canonical_logits = self.canonical_classifier(sequence)
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
        weights = attention_mask.to(sequence.dtype).unsqueeze(-1)
        masked_mean = (sequence * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        document_representation = 0.5 * (sequence[:, 0] + masked_mean)
        document_logits = self.document_classifier(document_representation)

        relation_logits = None
        if relation_source_masks is not None and relation_target_masks is not None:
            if relation_geometry is None or relation_source_types is None or relation_target_types is None:
                raise ValueError("all relation tensors are required when relation masks are provided")
            source_embeddings = self._pool_spans(sequence, relation_source_masks)
            target_embeddings = self._pool_spans(sequence, relation_target_masks)
            relation_logits = self.relation_head(
                source_embeddings,
                target_embeddings,
                relation_geometry.to(sequence.dtype),
                relation_source_types,
                relation_target_types,
            )

        task_losses: dict[str, torch.Tensor] = {}
        entity_loss = self._masked_cross_entropy(
            entity_logits,
            entity_labels,
            class_weights=getattr(self.config, "entity_class_weights", None),
        )
        if entity_loss is not None:
            task_losses["entity"] = entity_loss
        document_loss = self._masked_cross_entropy(
            document_logits,
            document_labels,
            class_weights=getattr(self.config, "document_class_weights", None),
        )
        if document_loss is not None:
            task_losses["document"] = document_loss
        canonical_loss = self._masked_cross_entropy(
            canonical_logits,
            canonical_labels,
            class_weights=getattr(self.config, "canonical_class_weights", None),
        )
        if canonical_loss is not None:
            task_losses["canonical"] = canonical_loss
        relation_loss = self._masked_cross_entropy(
            relation_logits,
            relation_labels,
            class_weights=getattr(self.config, "relation_class_weights", None),
        )
        if relation_loss is not None:
            task_losses["relation"] = relation_loss

        loss = None
        if task_losses:
            loss = (
                task_losses.get("entity", sequence.sum() * 0.0)
                + 0.2 * task_losses.get("document", sequence.sum() * 0.0)
                + 0.8 * task_losses.get("canonical", sequence.sum() * 0.0)
                + 0.6 * task_losses.get("relation", sequence.sum() * 0.0)
            )
        return MultiTaskTextLayoutOutput(
            loss=loss,
            entity_logits=entity_logits,
            document_logits=document_logits,
            canonical_logits=canonical_logits,
            relation_logits=relation_logits,
            task_losses=task_losses,
            hidden_states=outputs.hidden_states,
        )

    @staticmethod
    def _pool_spans(sequence: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        weights = masks.to(sequence.dtype)
        return torch.einsum("bps,bsh->bph", weights, sequence) / weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0)

    @staticmethod
    def _masked_cross_entropy(
        logits: Optional[torch.Tensor],
        labels: Optional[torch.LongTensor],
        *,
        class_weights: Optional[list[float]] = None,
    ) -> Optional[torch.Tensor]:
        if logits is None or labels is None:
            return None
        flat_labels = labels.reshape(-1)
        valid = flat_labels != -100
        if not bool(valid.any()):
            return None
        flat_logits = logits.reshape(-1, logits.shape[-1])
        weight_tensor = None
        if class_weights is not None:
            if len(class_weights) != logits.shape[-1]:
                raise ValueError("class weight count does not match logits")
            weight_tensor = torch.tensor(
                class_weights, dtype=logits.dtype, device=logits.device
            )
        return nn.functional.cross_entropy(
            flat_logits[valid], flat_labels[valid], weight=weight_tensor
        )
