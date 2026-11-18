"""Document ingestion package for KnowledgePilot AI (Phase 2)."""

from __future__ import annotations

from knowledgepilot.ingestion.chunker import TextChunker
from knowledgepilot.ingestion.errors import (
    EmptyDocumentError,
    IngestionError,
    MalformedDocumentError,
    UnsupportedFileTypeError,
)
from knowledgepilot.ingestion.loaders import (
    BaseLoader,
    PdfLoader,
    TxtLoader,
    load_document,
)
from knowledgepilot.ingestion.models import (
    Chunk,
    ChunkMetadata,
    Document,
    DocumentMetadata,
)

__all__ = [
    "BaseLoader",
    "Chunk",
    "ChunkMetadata",
    "Document",
    "DocumentMetadata",
    "EmptyDocumentError",
    "IngestionError",
    "MalformedDocumentError",
    "PdfLoader",
    "TextChunker",
    "TxtLoader",
    "UnsupportedFileTypeError",
    "load_document",
]
