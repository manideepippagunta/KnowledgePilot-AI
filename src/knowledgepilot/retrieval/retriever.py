"""High-level semantic and hybrid retriever combining dense vectors and BM25."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from knowledgepilot.embeddings.base import BaseEmbeddingModel
from knowledgepilot.ingestion.chunker import TextChunker
from knowledgepilot.ingestion.models import Chunk, Document
from knowledgepilot.retrieval.bm25 import BM25Index, reciprocal_rank_fusion
from knowledgepilot.retrieval.index import VectorIndex
from knowledgepilot.retrieval.models import SearchResult

RetrievalMode = Literal["dense", "bm25", "hybrid"]


class SemanticRetriever:
    """End-to-end semantic & hybrid search engine for ingested documents and chunks."""

    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        index: VectorIndex | None = None,
        bm25_index: BM25Index | None = None,
    ) -> None:
        """Initialize the retriever.

        Args:
            embedding_model: Embedding provider conforming to BaseEmbeddingModel.
            index: Optional pre-existing VectorIndex. If None, a new empty index
                is created matching the embedding model's dimension.
            bm25_index: Optional pre-existing BM25Index. If None, a new index
                is created and synchronized with `index` entries.
        """
        self.embedding_model = embedding_model
        self.index = index if index is not None else VectorIndex()
        self.bm25_index = bm25_index if bm25_index is not None else BM25Index()

        # If initialized with a pre-populated VectorIndex, sync BM25 automatically
        if bm25_index is None and len(self.index) > 0:
            existing_chunks = [chunk for chunk, _ in self.index._entries]
            self.bm25_index.add_batch(existing_chunks)

    def index_chunks(self, chunks: Sequence[Chunk]) -> int:
        """Embed and index a collection of Chunks across dense and BM25 stores."""
        if not chunks:
            return 0

        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_model.embed_texts(texts)
        self.index.add_batch(chunks, embeddings)
        self.bm25_index.add_batch(chunks)
        return len(chunks)

    def index_documents(
        self,
        documents: Sequence[Document],
        chunker: TextChunker | None = None,
    ) -> int:
        """Chunk, embed, and index a collection of Documents."""
        if not documents:
            return 0

        active_chunker = chunker if chunker is not None else TextChunker()
        chunks = active_chunker.split_documents(list(documents))
        return self.index_chunks(chunks)

    def search(
        self,
        query: str,
        top_k: int = 3,
        mode: RetrievalMode = "dense",
    ) -> list[SearchResult]:
        """Perform dense, BM25, or hybrid (RRF) search for the query string.

        Args:
            query: User search query.
            top_k: Number of most relevant results to return (must be > 0).
            mode: Retrieval mode ('dense', 'bm25', or 'hybrid'). Defaults to
                'dense' for backwards-compatible cosine similarity scoring.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty or whitespace-only.")

        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        if len(self.index) == 0:
            return []

        cleaned = query.strip()

        if mode == "bm25":
            return self.bm25_index.search(cleaned, top_k=top_k)

        query_vector = self.embedding_model.embed_text(cleaned)
        if mode == "dense":
            return self.index.search(query_vector, top_k=top_k)

        if mode == "hybrid":
            candidate_k = max(top_k * 3, 10)
            dense_results = self.index.search(query_vector, top_k=candidate_k)
            bm25_results = self.bm25_index.search(cleaned, top_k=candidate_k)
            return reciprocal_rank_fusion(
                [dense_results, bm25_results],
                k=60,
                top_k=top_k,
            )

        raise ValueError(f"Unsupported retrieval mode: {mode!r}")

    def search_hybrid(self, query: str, top_k: int = 3) -> list[SearchResult]:
        """Convenience wrapper for hybrid BM25 + dense RRF retrieval."""
        return self.search(query=query, top_k=top_k, mode="hybrid")
