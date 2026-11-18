"""Embedding models package for KnowledgePilot AI."""

from __future__ import annotations

from knowledgepilot.embeddings.base import BaseEmbeddingModel
from knowledgepilot.embeddings.mock import MockEmbeddingModel
from knowledgepilot.embeddings.sentence_transformer import (
    SentenceTransformerEmbedding,
)

__all__ = [
    "BaseEmbeddingModel",
    "MockEmbeddingModel",
    "SentenceTransformerEmbedding",
]
