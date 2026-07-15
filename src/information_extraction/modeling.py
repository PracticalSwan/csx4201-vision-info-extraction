"""Lightweight geometry-aware relation head for LayoutXLM entity spans."""
from __future__ import annotations

import torch
from torch import nn


class GeometryAwareRelationHead(nn.Module):
    """Classify typed entity pairs from span embeddings and geometry."""

    def __init__(
        self,
        hidden_size: int,
        geometry_size: int = 8,
        entity_type_count: int = 10,
        entity_type_embedding_size: int = 16,
        relation_type_count: int = 5,
    ) -> None:
        super().__init__()
        self.entity_type_embeddings = nn.Embedding(entity_type_count, entity_type_embedding_size)
        input_size = hidden_size * 2 + geometry_size + entity_type_embedding_size * 2
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, relation_type_count),
        )

    def forward(
        self,
        source_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
        geometry_features: torch.Tensor,
        source_types: torch.Tensor,
        target_types: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat([
            source_embeddings, target_embeddings, geometry_features,
            self.entity_type_embeddings(source_types), self.entity_type_embeddings(target_types),
        ], dim=-1)
        return self.classifier(features)
