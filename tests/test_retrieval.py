"""Unit tests for Phase 3 embeddings, vector index, and semantic retrieval."""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest

from knowledgepilot.embeddings import (
    MockEmbeddingModel,
    SentenceTransformerEmbedding,
)
from knowledgepilot.ingestion import (
    Chunk,
    ChunkMetadata,
    Document,
    DocumentMetadata,
)
from knowledgepilot.retrieval import (
    SearchResult,
    SemanticRetriever,
    VectorIndex,
)

# ---------------------------------------------------------------------------
# Test Helpers & Fixtures
# ---------------------------------------------------------------------------


def _make_chunk(
    text: str,
    file_name: str = "doc.txt",
    page_number: int | None = 1,
    chunk_index: int = 0,
) -> Chunk:
    """Create a test Chunk with standard metadata."""
    meta = ChunkMetadata(
        source=f"/data/{file_name}",
        file_name=file_name,
        file_type=file_name.split(".")[-1],
        page_number=page_number,
        chunk_index=chunk_index,
        start_char=0,
        end_char=len(text),
    )
    return Chunk(text=text, metadata=meta)


# ---------------------------------------------------------------------------
# 1. BaseEmbeddingModel & MockEmbeddingModel Tests
# ---------------------------------------------------------------------------


def test_mock_embedding_invalid_dimension() -> None:
    with pytest.raises(ValueError, match="dimension must be positive"):
        MockEmbeddingModel(dimension=0)
    with pytest.raises(ValueError, match="dimension must be positive"):
        MockEmbeddingModel(dimension=-5)


def test_mock_embedding_custom_vector_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="has length 2, expected 3"):
        MockEmbeddingModel(dimension=3, vectors={"hello": [1.0, 0.0]})


def test_mock_embedding_empty_text_raises() -> None:
    embedder = MockEmbeddingModel(dimension=4)
    with pytest.raises(ValueError, match="Cannot embed empty"):
        embedder.embed_text("")
    with pytest.raises(ValueError, match="Cannot embed empty"):
        embedder.embed_text("   \n\t  ")


def test_mock_embedding_empty_texts_batch() -> None:
    embedder = MockEmbeddingModel(dimension=4)
    assert embedder.embed_texts([]) == []


def test_mock_embedding_batch_with_invalid_text_raises() -> None:
    embedder = MockEmbeddingModel(dimension=4)
    with pytest.raises(ValueError, match="Text at index 1 is empty"):
        embedder.embed_texts(["valid text", "  ", "another text"])


def test_mock_embedding_unit_normalized() -> None:
    embedder = MockEmbeddingModel(dimension=4)
    vec = embedder.embed_text("KnowledgePilot retrieval")
    assert len(vec) == 4
    norm = math.sqrt(sum(x * x for x in vec))
    assert pytest.approx(norm, rel=1e-5) == 1.0


def test_mock_embedding_preset_vectors() -> None:
    presets = {
        "apple": [1.0, 0.0, 0.0],
        "banana": [0.0, 1.0, 0.0],
    }
    embedder = MockEmbeddingModel(dimension=3, vectors=presets)
    assert embedder.embed_text("apple") == [1.0, 0.0, 0.0]
    assert embedder.embed_text("banana") == [0.0, 1.0, 0.0]


# ---------------------------------------------------------------------------
# 2. SentenceTransformerEmbedding Lazy Loading & Interface Tests
# ---------------------------------------------------------------------------


def test_sentence_transformer_lazy_loading() -> None:
    # Creating instance must NOT import or load the model
    provider = SentenceTransformerEmbedding(model_name="all-MiniLM-L6-v2")
    assert provider._model is None
    # Known dimension should be accessible without loading model
    assert provider.dimension == 384
    assert provider._model is None


def test_sentence_transformer_empty_text_validation() -> None:
    provider = SentenceTransformerEmbedding()
    with pytest.raises(ValueError, match="Cannot embed empty"):
        provider.embed_text("   ")
    assert provider.embed_texts([]) == []
    with pytest.raises(ValueError, match="Text at index 0 is empty"):
        provider.embed_texts(["", "valid text"])


def test_sentence_transformer_mocked_inference() -> None:
    provider = SentenceTransformerEmbedding(model_name="all-MiniLM-L6-v2")

    mock_model = MagicMock()
    mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
    mock_model.get_sentence_embedding_dimension.return_value = 3

    with patch.object(provider, "_get_model", return_value=mock_model):
        result = provider.embed_text("Hello test")
        assert result == [0.1, 0.2, 0.3]
        mock_model.encode.assert_called_once()


def test_sentence_transformer_import_error_wrapped() -> None:
    provider = SentenceTransformerEmbedding()
    with (
        patch.dict("sys.modules", {"sentence_transformers": None}),
        pytest.raises(RuntimeError, match="sentence-transformers is required"),
    ):
        provider._get_model()


# ---------------------------------------------------------------------------
# 3. VectorIndex Unit Tests
# ---------------------------------------------------------------------------


def test_vector_index_empty_state() -> None:
    index = VectorIndex(dimension=3)
    assert len(index) == 0
    assert index.dimension == 3
    # Searching empty index returns empty list
    assert index.search([1.0, 0.0, 0.0], top_k=5) == []


def test_vector_index_invalid_dimension_raises() -> None:
    with pytest.raises(ValueError, match="dimension must be positive"):
        VectorIndex(dimension=0)
    with pytest.raises(ValueError, match="dimension must be positive"):
        VectorIndex(dimension=-2)


def test_vector_index_infer_dimension_from_first_add() -> None:
    index = VectorIndex()
    assert index.dimension is None

    chunk = _make_chunk("First chunk")
    index.add(chunk, [1.0, 2.0, 3.0])
    assert index.dimension == 3
    assert len(index) == 1


def test_vector_index_empty_embedding_raises() -> None:
    index = VectorIndex()
    chunk = _make_chunk("Chunk")
    with pytest.raises(ValueError, match="Embedding vector cannot be empty"):
        index.add(chunk, [])


def test_vector_index_dimension_mismatch_on_add() -> None:
    index = VectorIndex(dimension=3)
    chunk = _make_chunk("Mismatch chunk")
    with pytest.raises(ValueError, match="Embedding dimension mismatch"):
        index.add(chunk, [1.0, 2.0])


def test_vector_index_add_batch_mismatch_lengths() -> None:
    index = VectorIndex(dimension=2)
    c1 = _make_chunk("C1")
    c2 = _make_chunk("C2")
    with pytest.raises(ValueError, match="does not match number of embeddings"):
        index.add_batch([c1, c2], [[1.0, 0.0]])


def test_vector_index_search_validation() -> None:
    index = VectorIndex(dimension=2)
    chunk = _make_chunk("Chunk")
    index.add(chunk, [1.0, 0.0])

    with pytest.raises(ValueError, match="top_k must be positive"):
        index.search([1.0, 0.0], top_k=0)

    with pytest.raises(ValueError, match="top_k must be positive"):
        index.search([1.0, 0.0], top_k=-1)

    with pytest.raises(ValueError, match="Query embedding cannot be empty"):
        index.search([], top_k=1)

    with pytest.raises(ValueError, match="Query embedding dimension mismatch"):
        index.search([1.0, 0.0, 0.0], top_k=1)


def test_vector_index_cosine_ranking() -> None:
    index = VectorIndex(dimension=2)

    c_exact = _make_chunk("Exact match", file_name="exact.txt")
    c_partial = _make_chunk("Partial match", file_name="partial.txt")
    c_opposite = _make_chunk("Opposite match", file_name="opposite.txt")

    # c_exact is parallel to [1, 0] -> cosine = 1.0
    # c_partial is at 45 deg [1, 1] -> cosine = 0.7071
    # c_opposite is orthogonal [0, 1] -> cosine = 0.0
    index.add(c_exact, [1.0, 0.0])
    index.add(c_partial, [1.0, 1.0])
    index.add(c_opposite, [0.0, 1.0])

    results = index.search([1.0, 0.0], top_k=3)
    assert len(results) == 3

    assert results[0].file_name == "exact.txt"
    assert pytest.approx(results[0].score, rel=1e-4) == 1.0

    assert results[1].file_name == "partial.txt"
    assert pytest.approx(results[1].score, rel=1e-4) == 0.7071

    assert results[2].file_name == "opposite.txt"
    assert pytest.approx(results[2].score, rel=1e-4) == 0.0


def test_vector_index_zero_vector_similarity() -> None:
    index = VectorIndex(dimension=2)
    chunk = _make_chunk("Zero chunk")
    index.add(chunk, [0.0, 0.0])

    results = index.search([1.0, 1.0], top_k=1)
    assert len(results) == 1
    assert results[0].score == 0.0


def test_vector_index_tie_handling_deterministic() -> None:
    index = VectorIndex(dimension=2)

    c_b = _make_chunk("Chunk B", file_name="b.txt", chunk_index=0)
    c_a = _make_chunk("Chunk A", file_name="a.txt", chunk_index=0)
    c_c = _make_chunk("Chunk C", file_name="c.txt", chunk_index=0)

    # Identical vectors -> tie in cosine similarity
    index.add(c_b, [1.0, 0.0])
    index.add(c_a, [1.0, 0.0])
    index.add(c_c, [1.0, 0.0])

    results = index.search([1.0, 0.0], top_k=3)
    assert len(results) == 3
    # Ties broken by file_name ascending: a.txt, b.txt, c.txt
    assert [r.file_name for r in results] == ["a.txt", "b.txt", "c.txt"]


def test_search_result_metadata_properties() -> None:
    chunk = Chunk(
        text="Special chunk text",
        metadata=ChunkMetadata(
            source="/docs/manual.pdf",
            file_name="manual.pdf",
            file_type="pdf",
            page_number=4,
            chunk_index=7,
            start_char=100,
            end_char=118,
            extra={"tag": "reference"},
        ),
    )
    result = SearchResult(chunk=chunk, score=0.95)

    assert result.text == "Special chunk text"
    assert result.source == "/docs/manual.pdf"
    assert result.file_name == "manual.pdf"
    assert result.file_type == "pdf"
    assert result.page_number == 4
    assert result.chunk_index == 7
    assert result.score == 0.95


# ---------------------------------------------------------------------------
# 4. SemanticRetriever End-to-End Tests
# ---------------------------------------------------------------------------


def test_semantic_retriever_empty_index_search() -> None:
    embedder = MockEmbeddingModel(dimension=2)
    retriever = SemanticRetriever(embedding_model=embedder)

    assert len(retriever.index) == 0
    results = retriever.search("python query", top_k=3)
    assert results == []


def test_semantic_retriever_invalid_query_raises() -> None:
    embedder = MockEmbeddingModel(dimension=2)
    retriever = SemanticRetriever(embedding_model=embedder)

    with pytest.raises(ValueError, match="Query string cannot be empty"):
        retriever.search("")
    with pytest.raises(ValueError, match="Query string cannot be empty"):
        retriever.search("   ")


def test_semantic_retriever_invalid_top_k_raises() -> None:
    embedder = MockEmbeddingModel(dimension=2)
    retriever = SemanticRetriever(embedding_model=embedder)

    with pytest.raises(ValueError, match="top_k must be positive"):
        retriever.search("valid query", top_k=0)


def test_semantic_retriever_index_chunks_empty() -> None:
    embedder = MockEmbeddingModel(dimension=2)
    retriever = SemanticRetriever(embedding_model=embedder)
    assert retriever.index_chunks([]) == 0


def test_semantic_retriever_index_documents_empty() -> None:
    embedder = MockEmbeddingModel(dimension=2)
    retriever = SemanticRetriever(embedding_model=embedder)
    assert retriever.index_documents([]) == 0


def test_semantic_retriever_end_to_end_ranking_with_metadata() -> None:
    # Use preset vectors for predictable similarity
    vectors = {
        "python async event loop": [1.0, 0.0, 0.0],
        "database relational postgres": [0.0, 1.0, 0.0],
        "css styles flexbox layout": [0.0, 0.0, 1.0],
    }
    embedder = MockEmbeddingModel(dimension=3, vectors=vectors)
    retriever = SemanticRetriever(embedding_model=embedder)

    doc1 = Document(
        text="python async event loop",
        metadata=DocumentMetadata(
            source="/docs/python.pdf",
            file_name="python.pdf",
            file_type="pdf",
            page_number=2,
            total_pages=5,
        ),
    )
    doc2 = Document(
        text="database relational postgres",
        metadata=DocumentMetadata(
            source="/docs/sql.txt",
            file_name="sql.txt",
            file_type="txt",
            page_number=1,
            total_pages=1,
        ),
    )
    doc3 = Document(
        text="css styles flexbox layout",
        metadata=DocumentMetadata(
            source="/docs/web.pdf",
            file_name="web.pdf",
            file_type="pdf",
            page_number=7,
            total_pages=10,
        ),
    )

    indexed_count = retriever.index_documents([doc1, doc2, doc3])
    assert indexed_count == 3
    assert len(retriever.index) == 3

    # Query targeting python
    embedder_query = MockEmbeddingModel(
        dimension=3,
        vectors={"python query": [1.0, 0.0, 0.0]},
    )
    retriever.embedding_model = embedder_query

    results = retriever.search("python query", top_k=2)
    assert len(results) == 2

    # Top match should be doc1 (python) with page 2
    top = results[0]
    assert top.file_name == "python.pdf"
    assert top.page_number == 2
    assert top.source == "/docs/python.pdf"
    assert "python async" in top.text
    assert pytest.approx(top.score, rel=1e-4) == 1.0

    # Second result should have 0 score because other vectors are orthogonal
    assert results[1].score == 0.0


def test_vector_index_serialization_export_and_import() -> None:
    """Verify VectorIndex.to_list() and load_entries() preserve entries and search."""
    index = VectorIndex()
    c1 = _make_chunk("Chunk one", file_name="doc1.txt", chunk_index=0)
    c2 = _make_chunk("Chunk two", file_name="doc2.txt", chunk_index=1)
    index.add(c1, [1.0, 0.0])
    index.add(c2, [0.0, 1.0])

    serialized = index.to_list()
    assert len(serialized) == 2
    assert serialized[0]["chunk"]["text"] == "Chunk one"
    assert serialized[0]["vector"] == [1.0, 0.0]

    # Restore into fresh index
    new_index = VectorIndex()
    new_index.load_entries(serialized)

    assert len(new_index) == 2
    assert new_index.dimension == 2
    results = new_index.search([1.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0].file_name == "doc1.txt"
    assert results[0].text == "Chunk one"
