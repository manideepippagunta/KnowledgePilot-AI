"""Semantic retrieval package for KnowledgePilot AI (Phase 3)."""

from __future__ import annotations

from knowledgepilot.retrieval.index import VectorIndex
from knowledgepilot.retrieval.models import SearchResult
from knowledgepilot.retrieval.retriever import SemanticRetriever

__all__ = [
    "SearchResult",
    "SemanticRetriever",
    "VectorIndex",
]
