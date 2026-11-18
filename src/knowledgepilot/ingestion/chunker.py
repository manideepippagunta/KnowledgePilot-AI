"""Text chunking utilities with overlap and metadata tracking."""

from __future__ import annotations

from knowledgepilot.ingestion.models import Chunk, ChunkMetadata, Document


class TextChunker:
    """Configurable text chunker with sliding window overlap.

    Produces text chunks and preserves character offsets and source metadata.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        """Initialize the chunker with target size and overlap.

        Args:
            chunk_size: Maximum number of characters in each chunk. Must be > 0.
            chunk_overlap: Number of characters shared between adjacent chunks.
                Must be >= 0 and strictly less than chunk_size.

        Raises:
            ValueError: If chunk_size <= 0, chunk_overlap < 0, or
                chunk_overlap >= chunk_size.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than "
                f"chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[tuple[str, int, int]]:
        """Split a string into slices of (chunk_text, start_char, end_char).

        Args:
            text: Raw input string to split.

        Returns:
            A list of tuples: (chunk_text, start_char_index, end_char_index).
            Returns an empty list if the text is empty or only whitespace.
        """
        if not text or not text.strip():
            return []

        text_len = len(text)
        step = self.chunk_size - self.chunk_overlap
        slices: list[tuple[str, int, int]] = []
        start = 0

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            slices.append((text[start:end], start, end))
            if end >= text_len:
                break
            start += step

        return slices

    def split_document(
        self, document: Document, *, start_chunk_index: int = 0
    ) -> list[Chunk]:
        """Split a single Document into Chunks, carrying over metadata.

        Args:
            document: The Document to split.
            start_chunk_index: Starting offset for the chunk_index sequence.

        Returns:
            A list of Chunk instances with updated chunk metadata.
        """
        slices = self.split_text(document.text)
        chunks: list[Chunk] = []

        for offset, (chunk_text, start_char, end_char) in enumerate(slices):
            chunk_idx = start_chunk_index + offset
            meta = ChunkMetadata(
                source=document.metadata.source,
                file_name=document.metadata.file_name,
                file_type=document.metadata.file_type,
                page_number=document.metadata.page_number,
                chunk_index=chunk_idx,
                start_char=start_char,
                end_char=end_char,
                extra=dict(document.metadata.extra),
            )
            chunks.append(Chunk(text=chunk_text, metadata=meta))

        return chunks

    def split_documents(self, documents: list[Document]) -> list[Chunk]:
        """Split multiple Documents into Chunks with sequential indexing.

        Args:
            documents: Sequence of Document instances (e.g. pages of a PDF).

        Returns:
            A concatenated list of Chunk instances with sequential chunk_index.
        """
        all_chunks: list[Chunk] = []
        current_idx = 0

        for doc in documents:
            doc_chunks = self.split_document(doc, start_chunk_index=current_idx)
            all_chunks.extend(doc_chunks)
            current_idx += len(doc_chunks)

        return all_chunks
