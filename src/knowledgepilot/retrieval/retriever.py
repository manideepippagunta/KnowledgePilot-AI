"""High-level semantic retriever combining text chunking and vector search."""

from __future__ import annotations

from collections.abc import Sequence

from knowledgepilot.embeddings.base import BaseEmbeddingModel
from knowledgepilot.ingestion.chunker import TextChunker
from knowledgepilot.ingestion.models import Chunk, Document
from knowledgepilot.retrieval.index import VectorIndex
from knowledgepilot.retrieval.models import SearchResult


class SemanticRetriever:
    """End-to-end semantic search engine for ingested documents and chunks."""

    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        index: VectorIndex | None = None,
    ) -> None:
        """Initialize the retriever.

        Args:
            embedding_model: Embedding provider conforming to BaseEmbeddingModel.
            index: Optional pre-existing VectorIndex. If None, a new empty index
                is created matching the embedding model's dimension.
        """
        self.embedding_model = embedding_model
        self.index = index if index is not None else VectorIndex()

    def index_chunks(self, chunks: Sequence[Chunk]) -> int:
        """Embed and index a collection of Chunks into the vector store."""
        if not chunks:
            return 0

        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_model.embed_texts(texts)
        self.index.add_batch(chunks, embeddings)
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
    ) -> list[SearchResult]:
        """Perform dense vector search for the query string.

        Args:
            query: User search query.
            top_k: Number of most relevant results to return (must be > 0).
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty or whitespace-only.")

        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        if len(self.index) == 0:
            return []

        cleaned = query.strip()
        query_vector = self.embedding_model.embed_text(cleaned)
        return self.index.search(query_vector, top_k=top_k)
