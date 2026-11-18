"""In-memory vector index with cosine-similarity search."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, is_dataclass
from typing import Any

from knowledgepilot.ingestion.models import Chunk, ChunkMetadata
from knowledgepilot.retrieval.models import SearchResult


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute cosine similarity between two float vectors.

    Returns 0.0 if either vector has zero magnitude. Clamps the result
    to [-1.0, 1.0] to guard against floating-point inaccuracies.
    """
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b, strict=True))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    raw = dot_product / (norm_a * norm_b)
    return max(-1.0, min(1.0, raw))


class VectorIndex:
    """In-memory vector store mapping text Chunks to dense embeddings."""

    def __init__(self, dimension: int | None = None) -> None:
        """Initialize the vector index.

        Args:
            dimension: Optional expected vector dimensionality. If None,
                it will be inferred from the first added embedding.

        Raises:
            ValueError: If dimension is specified and <= 0.
        """
        if dimension is not None and dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")

        self._dimension: int | None = dimension
        self._entries: list[tuple[Chunk, list[float]]] = []

    @property
    def dimension(self) -> int | None:
        """The dimensionality of embeddings stored in this index, if known."""
        return self._dimension

    def __len__(self) -> int:
        """Return the number of chunks stored in the index."""
        return len(self._entries)

    @property
    def total_chunks(self) -> int:
        """Return the total number of chunks indexed."""
        return len(self._entries)

    def add(self, chunk: Chunk, embedding: Sequence[float]) -> None:
        """Add a single Chunk and its embedding to the index.

        Args:
            chunk: The Chunk to store.
            embedding: Dense float embedding vector.

        Raises:
            ValueError: If embedding is empty, or dimension does not match.
        """
        if not embedding:
            raise ValueError("Embedding vector cannot be empty.")

        emb_len = len(embedding)
        if self._dimension is None:
            self._dimension = emb_len
        elif emb_len != self._dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._dimension}, "
                f"got {emb_len}."
            )

        self._entries.append((chunk, [float(x) for x in embedding]))

    def add_batch(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Add multiple chunks and their corresponding embeddings in batch.

        Args:
            chunks: Sequence of Chunk instances.
            embeddings: Sequence of dense float embedding vectors.

        Raises:
            ValueError: If counts mismatch or any embedding has invalid dimension.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Number of chunks ({len(chunks)}) does not match "
                f"number of embeddings ({len(embeddings)})."
            )

        for chunk, emb in zip(chunks, embeddings, strict=True):
            self.add(chunk, emb)

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 3,
    ) -> list[SearchResult]:
        """Search for the most similar chunks to the query vector.

        Args:
            query_embedding: Dense query vector.
            top_k: Maximum number of results to return (must be > 0).

        Returns:
            A list of SearchResult objects ordered by similarity descending.
            Ties are broken deterministically by chunk_index and filename.

        Raises:
            ValueError: If top_k <= 0, query_embedding is empty, or query
                dimension does not match index dimension.
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        if len(self._entries) == 0:
            return []

        if not query_embedding:
            raise ValueError("Query embedding cannot be empty.")

        if self._dimension is not None and len(query_embedding) != self._dimension:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {self._dimension}, "
                f"got {len(query_embedding)}."
            )

        scored: list[SearchResult] = []
        for chunk, emb in self._entries:
            sim = _cosine_similarity(query_embedding, emb)
            scored.append(SearchResult(chunk=chunk, score=sim))

        # Deterministic sorting: highest score first;
        # tie-breakers: file_name, chunk_index, text
        scored.sort(
            key=lambda r: (
                -r.score,
                r.chunk.metadata.file_name,
                r.chunk.metadata.chunk_index,
                r.chunk.text,
            )
        )

        return scored[:top_k]

    def to_list(self) -> list[dict[str, Any]]:
        """Serialize index entries into JSON-serializable list of dicts."""
        return vector_index_export_entries(self)

    def load_entries(
        self,
        raw_entries: list[dict[str, Any]],
        chunk_cls: type[Any] = Chunk,
    ) -> None:
        """Restore index entries from a serialized list of dicts."""
        vector_index_import_entries(self, raw_entries, chunk_cls=chunk_cls)


def vector_index_export_entries(index: Any) -> list[dict[str, Any]]:
    """Serialize in-memory `(Chunk, vector)` entries into JSON-safe dictionaries."""
    serialized: list[dict[str, Any]] = []
    entries = getattr(index, "_entries", [])
    for chunk, vector in entries:
        dump_fn = getattr(chunk, "model_dump", None)
        if callable(dump_fn):
            chunk_dict = dump_fn()
        elif is_dataclass(chunk) and not isinstance(chunk, type):
            chunk_dict = asdict(chunk)
        else:
            chunk_dict = dict(chunk.__dict__)
        serialized.append(
            {
                "chunk": chunk_dict,
                "vector": [float(v) for v in vector],
            }
        )
    return serialized


def vector_index_import_entries(
    index: Any,
    raw_entries: list[dict[str, Any]],
    chunk_cls: type[Any] = Chunk,
) -> None:
    """Restore `(Chunk, vector)` entries into `VectorIndex` without re-embedding."""
    restored: list[tuple[Any, list[float]]] = []
    for item in raw_entries:
        chunk_data = dict(item["chunk"])
        meta_data = chunk_data.get("metadata")
        if isinstance(meta_data, dict):
            chunk_data["metadata"] = ChunkMetadata(**meta_data)
        chunk_obj = chunk_cls(**chunk_data)
        vec = [float(x) for x in item["vector"]]
        restored.append((chunk_obj, vec))

    index._entries = restored
    if restored and getattr(index, "_dimension", None) is None:
        index._dimension = len(restored[0][1])
