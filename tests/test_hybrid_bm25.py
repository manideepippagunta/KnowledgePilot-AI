"""Tests for BM25 sparse search, Reciprocal Rank Fusion (RRF), and eval benchmark."""

from __future__ import annotations

from knowledgepilot.embeddings.mock import MockEmbeddingModel
from knowledgepilot.ingestion.models import Chunk, ChunkMetadata
from knowledgepilot.retrieval.bm25 import BM25Index, reciprocal_rank_fusion, tokenize
from knowledgepilot.retrieval.retriever import SemanticRetriever


def _make_chunk(text: str, file_name: str, chunk_index: int) -> Chunk:
    return Chunk(
        text=text,
        metadata=ChunkMetadata(
            source=file_name,
            file_name=file_name,
            file_type="txt",
            page_number=1,
            chunk_index=chunk_index,
            start_char=0,
            end_char=len(text),
        ),
    )


def test_bm25_exact_keyword_and_acronym_matching() -> None:
    index = BM25Index()
    chunks = [
        _make_chunk("AES-256 encryption and SOC 2 Type II compliance.", "sec.txt", 0),
        _make_chunk("Kubernetes clusters and Terraform deployments.", "infra.txt", 1),
    ]
    index.add_batch(chunks)

    results = index.search("SOC 2 encryption", top_k=2)
    assert len(results) == 1
    assert results[0].chunk.metadata.file_name == "sec.txt"
    assert results[0].score > 0.0


def test_reciprocal_rank_fusion_combines_ranked_lists() -> None:
    c1 = _make_chunk("Chunk one about Chaos Monkey", "a.txt", 0)
    c2 = _make_chunk("Chunk two about Kubernetes", "b.txt", 1)

    dense = BM25Index()
    dense.add_batch([c1, c2])
    r1 = dense.search("Chaos Monkey", top_k=2)
    r2 = dense.search("Chaos", top_k=2)

    fused = reciprocal_rank_fusion([r1, r2], k=60, top_k=2)
    assert len(fused) == 1
    assert fused[0].chunk.metadata.file_name == "a.txt"


def test_semantic_retriever_hybrid_and_bm25_modes() -> None:
    retriever = SemanticRetriever(embedding_model=MockEmbeddingModel(dimension=32))
    chunks = [
        _make_chunk("ISO-27001 requires quarterly audits.", "compliance.txt", 0),
        _make_chunk("Multi-region replication across us-east-1.", "cloud.txt", 1),
    ]
    retriever.index_chunks(chunks)

    bm25_res = retriever.search("ISO-27001 audits", top_k=1, mode="bm25")
    assert len(bm25_res) == 1
    assert bm25_res[0].chunk.metadata.file_name == "compliance.txt"

    hybrid_res = retriever.search_hybrid("ISO-27001 audits", top_k=2)
    assert len(hybrid_res) >= 1
    assert hybrid_res[0].chunk.metadata.file_name == "compliance.txt"


def test_tokenize_handles_hyphenated_identifiers() -> None:
    tokens = tokenize("ISO-27001 AES-256 us-east-1")
    assert tokens == ["iso-27001", "aes-256", "us-east-1"]
