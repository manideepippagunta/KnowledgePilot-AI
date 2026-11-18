"""Data models for document ingestion and text chunking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class DocumentMetadata:
    """Metadata describing an ingested document or page.

    Attributes:
        source: Absolute or relative path / URI of the source file.
        file_name: Base filename including extension.
        file_type: Normalized file type identifier (e.g. 'txt', 'pdf').
        page_number: 1-based page number for paginated formats, or None.
        total_pages: Total number of pages in the source file, or None.
        extra: Additional arbitrary metadata attributes.
    """

    source: str
    file_name: str
    file_type: str
    page_number: int | None = None
    total_pages: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Document:
    """An ingested piece of content (document or page) with associated metadata.

    Attributes:
        text: Raw text content extracted from the document or page.
        metadata: Associated document metadata.
    """

    text: str
    metadata: DocumentMetadata


@dataclass(frozen=True, slots=True)
class ChunkMetadata:
    """Metadata describing a chunk of text derived from a document.

    Attributes:
        source: Source file path or URI.
        file_name: Base filename including extension.
        file_type: Normalized file type identifier (e.g. 'txt', 'pdf').
        page_number: 1-based page number from the source document, or None.
        chunk_index: Monotonically increasing index within the sequence of chunks.
        start_char: Zero-based starting character offset within the source text.
        end_char: Zero-based ending character offset within the source text.
        extra: Additional arbitrary metadata attributes.
    """

    source: str
    file_name: str
    file_type: str
    page_number: int | None = None
    chunk_index: int = 0
    start_char: int = 0
    end_char: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Chunk:
    """A chunk of text ready for embedding or retrieval.

    Attributes:
        text: Chunk text content.
        metadata: Associated chunk metadata.
    """

    text: str
    metadata: ChunkMetadata
