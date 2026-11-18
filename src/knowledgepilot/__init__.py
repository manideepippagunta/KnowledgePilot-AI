"""KnowledgePilot AI — RAG application foundation."""

from __future__ import annotations

from knowledgepilot.config import Settings
from knowledgepilot.embeddings import (
    BaseEmbeddingModel,
    MockEmbeddingModel,
    SentenceTransformerEmbedding,
)
from knowledgepilot.generation import (
    BaseGenerator,
    GenerationError,
    MockGenerator,
    OllamaGenerator,
    create_generator,
)
from knowledgepilot.ingestion import (
    Chunk,
    ChunkMetadata,
    Document,
    DocumentMetadata,
    TextChunker,
    load_document,
)
from knowledgepilot.rag import (
    Citation,
    RagAnswer,
    RagService,
    ask_document,
)
from knowledgepilot.retrieval import (
    SearchResult,
    SemanticRetriever,
    VectorIndex,
)

__all__ = [
    "BaseEmbeddingModel",
    "BaseGenerator",
    "Chunk",
    "ChunkMetadata",
    "Citation",
    "Document",
    "DocumentMetadata",
    "GenerationError",
    "MockEmbeddingModel",
    "MockGenerator",
    "OllamaGenerator",
    "RagAnswer",
    "RagService",
    "SearchResult",
    "SemanticRetriever",
    "SentenceTransformerEmbedding",
    "Settings",
    "TextChunker",
    "VectorIndex",
    "ask_document",
    "create_generator",
    "load_document",
]
__version__ = "0.1.0"
