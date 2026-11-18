"""Data models for semantic retrieval results."""

from __future__ import annotations

from dataclasses import dataclass

from knowledgepilot.ingestion.models import Chunk


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A retrieved chunk match with similarity score and metadata accessors.

    Attributes:
        chunk: The retrieved Chunk instance containing text and metadata.
        score: Cosine similarity score in range [-1.0, 1.0].
    """

    chunk: Chunk
    score: float

    @property
    def text(self) -> str:
        """The text content of the retrieved chunk."""
        return self.chunk.text

    @property
    def source(self) -> str:
        """Source filepath or URI of the document."""
        return self.chunk.metadata.source

    @property
    def file_name(self) -> str:
        """Original filename of the source document."""
        return self.chunk.metadata.file_name

    @property
    def file_type(self) -> str:
        """File type identifier (e.g. 'pdf', 'txt')."""
        return self.chunk.metadata.file_type

    @property
    def page_number(self) -> int | None:
        """1-based page number if available (e.g. for PDFs), else None."""
        return self.chunk.metadata.page_number

    @property
    def chunk_index(self) -> int:
        """Index of the chunk in the document sequence."""
        return self.chunk.metadata.chunk_index
