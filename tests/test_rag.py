"""Unit tests for Phase 4 grounded question answering and RAG service."""

from __future__ import annotations

import pytest

from knowledgepilot.embeddings import MockEmbeddingModel
from knowledgepilot.generation import (
    GenerationError,
    MockGenerator,
    OllamaGenerator,
    format_grounded_qa_prompt,
)
from knowledgepilot.ingestion import Chunk, ChunkMetadata
from knowledgepilot.rag import (
    INSUFFICIENT_EVIDENCE_ANSWER,
    Citation,
    RagAnswer,
    RagService,
)
from knowledgepilot.retrieval import SearchResult, SemanticRetriever, VectorIndex

# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------


def _make_search_result(
    text: str,
    file_name: str = "doc.pdf",
    page_number: int | None = 1,
    score: float = 0.9,
    chunk_index: int = 0,
) -> SearchResult:
    chunk = Chunk(
        text=text,
        metadata=ChunkMetadata(
            source=f"/data/{file_name}",
            file_name=file_name,
            file_type=file_name.split(".")[-1],
            page_number=page_number,
            chunk_index=chunk_index,
            start_char=0,
            end_char=len(text),
        ),
    )
    return SearchResult(chunk=chunk, score=score)


# ---------------------------------------------------------------------------
# 1. MockGenerator Unit Tests
# ---------------------------------------------------------------------------


def test_mock_generator_empty_prompt_raises() -> None:
    gen = MockGenerator()
    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        gen.generate("")
    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        gen.generate("   \n\t  ")


def test_mock_generator_default_response() -> None:
    gen = MockGenerator(default_response="Default fact.")
    assert gen.generate("What is the speed of light?") == "Default fact."
    assert len(gen.calls) == 1


def test_mock_generator_keyword_responses() -> None:
    gen = MockGenerator(
        responses={"python": "Python is a programming language."},
        default_response="Fallback.",
    )
    assert (
        gen.generate("Tell me about python features")
        == "Python is a programming language."
    )
    assert gen.generate("Tell me about ruby") == "Fallback."


def test_mock_generator_custom_fn() -> None:
    gen = MockGenerator(custom_fn=lambda p: f"ECHO: {p[:10]}")
    assert gen.generate("KnowledgePilot AI") == "ECHO: KnowledgeP"


def test_mock_generator_failure_simulation() -> None:
    gen = MockGenerator(should_fail=True, error_message="API Rate limit")
    with pytest.raises(GenerationError, match="API Rate limit"):
        gen.generate("Any prompt")


# ---------------------------------------------------------------------------
# 2. Prompt Formatter Unit Tests
# ---------------------------------------------------------------------------


def test_format_prompt_empty_query_raises() -> None:
    with pytest.raises(ValueError, match="Query string cannot be empty"):
        format_grounded_qa_prompt("", [])
    with pytest.raises(ValueError, match="Query string cannot be empty"):
        format_grounded_qa_prompt("   ", [])


def test_format_prompt_evidence_grounding_and_citations() -> None:
    ev1 = _make_search_result(
        "RAG combines retrieval with generation.",
        file_name="rag.pdf",
        page_number=2,
    )
    ev2 = _make_search_result(
        "Vectors are stored in memory.",
        file_name="index.txt",
        page_number=None,
    )

    prompt = format_grounded_qa_prompt("How does RAG work?", [ev1, ev2])

    # Check system instructions
    assert "Answer the user's question using ONLY the directly stated facts" in prompt
    assert (
        "I do not have sufficient information in the provided documents "
        "to answer this question"
    ) in prompt

    # Check evidence structure
    assert "[Document 1] (File: rag.pdf, Page: 2)" in prompt
    assert "RAG combines retrieval with generation." in prompt
    assert "[Document 2] (File: index.txt, Page: N/A)" in prompt
    assert "Vectors are stored in memory." in prompt

    # Check query position
    assert "Question: How does RAG work?" in prompt
    assert prompt.endswith("Answer:")


def test_format_prompt_custom_system_instruction() -> None:
    ev = _make_search_result("Test context", file_name="test.txt")
    custom_inst = "Special custom instructions for enterprise security."
    prompt = format_grounded_qa_prompt(
        "Question?", [ev], system_instruction=custom_inst
    )
    assert prompt.startswith(custom_inst)


# ---------------------------------------------------------------------------
# 3. Models (Citation & RagAnswer) Tests
# ---------------------------------------------------------------------------


def test_citation_and_rag_answer_immutability() -> None:
    c = Citation(file_name="doc.pdf", page_number=5, source="/p/doc.pdf", chunk_index=0)
    ev = _make_search_result("Context", file_name="doc.pdf", page_number=5)
    ans = RagAnswer(
        query="What?",
        answer="The answer.",
        citations=[c],
        evidence=[ev],
        has_sufficient_evidence=True,
    )

    assert ans.query == "What?"
    assert ans.answer == "The answer."
    assert ans.has_sufficient_evidence is True
    assert ans.cited_files == ["doc.pdf"]
    assert ans.cited_pages == [("doc.pdf", 5)]

    with pytest.raises(AttributeError):
        ans.answer = "New answer"  # type: ignore[misc]

    with pytest.raises(AttributeError):
        c.page_number = 10  # type: ignore[misc]


def test_rag_answer_unique_cited_files_and_pages() -> None:
    c1 = Citation(
        file_name="doc.pdf", page_number=2, source="/p/doc.pdf", chunk_index=0
    )
    c2 = Citation(
        file_name="doc.pdf", page_number=2, source="/p/doc.pdf", chunk_index=1
    )
    c3 = Citation(
        file_name="doc.pdf", page_number=3, source="/p/doc.pdf", chunk_index=2
    )
    c4 = Citation(
        file_name="notes.txt",
        page_number=None,
        source="/p/notes.txt",
        chunk_index=3,
    )

    ans = RagAnswer(
        query="Q",
        answer="A",
        citations=[c1, c2, c3, c4],
        evidence=[],
    )

    assert ans.cited_files == ["doc.pdf", "notes.txt"]
    assert ans.cited_pages == [
        ("doc.pdf", 2),
        ("doc.pdf", 3),
        ("notes.txt", None),
    ]


# ---------------------------------------------------------------------------
# 4. RagService Integration & Orchestration Tests
# ---------------------------------------------------------------------------


def test_rag_service_query_validation() -> None:
    embedder = MockEmbeddingModel(dimension=2)
    retriever = SemanticRetriever(embedding_model=embedder)
    generator = MockGenerator()
    service = RagService(retriever=retriever, generator=generator)

    with pytest.raises(ValueError, match="Query string cannot be empty"):
        service.answer_question("")

    with pytest.raises(ValueError, match="Query string cannot be empty"):
        service.answer_question("   ")

    with pytest.raises(ValueError, match="top_k must be positive"):
        service.answer_question("Valid query", top_k=0)


def test_rag_service_empty_retrieval_returns_insufficient_evidence() -> None:
    embedder = MockEmbeddingModel(dimension=2)
    retriever = SemanticRetriever(embedding_model=embedder)
    generator = MockGenerator(default_response="Should not be called")
    service = RagService(retriever=retriever, generator=generator)

    result = service.answer_question("What is the project name?", top_k=3)

    assert result.query == "What is the project name?"
    assert result.answer == INSUFFICIENT_EVIDENCE_ANSWER
    assert result.has_sufficient_evidence is False
    assert result.citations == []
    assert result.evidence == []
    # Generator must not be called when no evidence exists
    assert len(generator.calls) == 0


def test_rag_service_grounded_answering_with_source_attribution() -> None:
    c1_text = "KnowledgePilot architecture uses local sentence transformers."
    c2_text = "General unrelated facts about space."
    vectors = {
        c1_text: [1.0, 0.0],
        c2_text: [0.0, 1.0],
    }
    embedder = MockEmbeddingModel(dimension=2, vectors=vectors)
    index = VectorIndex(dimension=2)
    retriever = SemanticRetriever(embedding_model=embedder, index=index)

    c1 = Chunk(
        text=c1_text,
        metadata=ChunkMetadata(
            source="/docs/architecture.pdf",
            file_name="architecture.pdf",
            file_type="pdf",
            page_number=4,
            chunk_index=0,
            start_char=0,
            end_char=len(c1_text),
        ),
    )
    c2 = Chunk(
        text=c2_text,
        metadata=ChunkMetadata(
            source="/docs/space.txt",
            file_name="space.txt",
            file_type="txt",
            page_number=1,
            chunk_index=1,
            start_char=0,
            end_char=len(c2_text),
        ),
    )
    retriever.index_chunks([c1, c2])

    expected_ans = (
        "KnowledgePilot architecture uses local sentence transformers "
        "as stated in [Document 1]."
    )
    generator = MockGenerator(default_response=expected_ans)
    service = RagService(retriever=retriever, generator=generator)

    embedder_query = MockEmbeddingModel(
        dimension=2,
        vectors={"architecture query": [1.0, 0.0]},
    )
    retriever.embedding_model = embedder_query

    result = service.answer_question("architecture query", top_k=1)

    assert result.has_sufficient_evidence is True
    assert result.answer == expected_ans
    assert len(result.citations) == 1
    cit = result.citations[0]
    assert cit.file_name == "architecture.pdf"
    assert cit.page_number == 4
    assert cit.source == "/docs/architecture.pdf"
    assert cit.chunk_index == 0

    assert result.cited_files == ["architecture.pdf"]
    assert result.cited_pages == [("architecture.pdf", 4)]

    # Generator received prompt with the exact chunk text
    assert len(generator.calls) == 1
    assert (
        "KnowledgePilot architecture uses local sentence transformers"
        in generator.calls[0]
    )


def test_rag_service_acknowledges_generator_insufficient_evidence() -> None:
    embedder = MockEmbeddingModel(dimension=2)
    index = VectorIndex(dimension=2)
    retriever = SemanticRetriever(embedding_model=embedder, index=index)

    chunk = Chunk(
        text="Some retrieved passage about weather.",
        metadata=ChunkMetadata(
            source="/docs/weather.pdf",
            file_name="weather.pdf",
            file_type="pdf",
            page_number=1,
            chunk_index=0,
            start_char=0,
            end_char=35,
        ),
    )
    retriever.index.add(chunk, [1.0, 0.0])

    # Generator states insufficient evidence
    generator = MockGenerator(
        default_response=(
            "I do not have sufficient information in the provided documents "
            "to answer this question."
        )
    )
    service = RagService(retriever=retriever, generator=generator)

    result = service.answer_question("What is quantum computing?", top_k=1)

    assert result.has_sufficient_evidence is False
    assert "do not have sufficient information" in result.answer


def test_rag_service_generator_failure_raises_generation_error() -> None:
    embedder = MockEmbeddingModel(dimension=2)
    retriever = SemanticRetriever(embedding_model=embedder)
    chunk = Chunk(
        text="Context",
        metadata=ChunkMetadata(
            source="/f.txt",
            file_name="f.txt",
            file_type="txt",
            page_number=1,
            chunk_index=0,
            start_char=0,
            end_char=7,
        ),
    )
    retriever.index.add(chunk, [1.0, 0.0])

    generator = MockGenerator(should_fail=True, error_message="Inference crashed")
    service = RagService(retriever=retriever, generator=generator)

    with pytest.raises(GenerationError, match="Inference crashed"):
        service.answer_question("Question")


def test_rag_service_unexpected_generator_failure_wrapped() -> None:
    embedder = MockEmbeddingModel(dimension=2)
    retriever = SemanticRetriever(embedding_model=embedder)
    chunk = Chunk(
        text="Context",
        metadata=ChunkMetadata(
            source="/f.txt",
            file_name="f.txt",
            file_type="txt",
            page_number=1,
            chunk_index=0,
            start_char=0,
            end_char=7,
        ),
    )
    retriever.index.add(chunk, [1.0, 0.0])

    def broken_gen(prompt: str) -> str:
        raise RuntimeError("Low-level memory corruption")

    generator = MockGenerator(custom_fn=broken_gen)
    service = RagService(retriever=retriever, generator=generator)

    with pytest.raises(GenerationError, match="Unexpected generation failure"):
        service.answer_question("Question")


# ---------------------------------------------------------------------------
# 5. Answer Grounding, Partial Evidence, and Citation Correctness Tests
# ---------------------------------------------------------------------------


def test_rag_partially_supported_question_distinguishes_missing_info() -> None:
    """Verify multi-phase pipelines where some phases are documented and others

    are missing: supported phases are cited, missing phases are explicitly acknowledged,
    and the answer is marked as grounded (has_sufficient_evidence=True).
    """
    chunk1_text = (
        "Netflix content processing: 1. Ingestion validates source master video "
        "mezzanines and extracts audio/video metadata."
    )
    chunk2_text = (
        "2. Transcoding converts video chunks into multi-bitrate streams using AV1, "
        "VP9, and H.264 codecs with parallel cloud workers."
    )
    vectors = {
        chunk1_text: [1.0, 0.0],
        chunk2_text: [0.0, 1.0],
    }
    embedder = MockEmbeddingModel(dimension=2, vectors=vectors)
    index = VectorIndex(dimension=2)
    retriever = SemanticRetriever(embedding_model=embedder, index=index)

    c1 = Chunk(
        text=chunk1_text,
        metadata=ChunkMetadata(
            source="/docs/netflix_pipeline.pdf",
            file_name="netflix_pipeline.pdf",
            file_type="pdf",
            page_number=3,
            chunk_index=0,
            start_char=0,
            end_char=len(chunk1_text),
        ),
    )
    c2 = Chunk(
        text=chunk2_text,
        metadata=ChunkMetadata(
            source="/docs/netflix_pipeline.pdf",
            file_name="netflix_pipeline.pdf",
            file_type="pdf",
            page_number=4,
            chunk_index=1,
            start_char=0,
            end_char=len(chunk2_text),
        ),
    )
    retriever.index_chunks([c1, c2])

    query = (
        "Describe Netflix's content pipeline: ingestion, transcoding, and packaging."
    )
    retriever.embedding_model = MockEmbeddingModel(
        dimension=2,
        vectors={query: [0.7, 0.7]},
    )

    grounded_answer = (
        "Based on the provided documents, the Netflix pipeline includes:\n"
        "- Ingestion: Validates source master video mezzanines and extracts "
        "audio/video metadata [Document 1, Page 3].\n"
        "- Transcoding: Converts video chunks into multi-bitrate streams using AV1, "
        "VP9, and H.264 codecs [Document 2, Page 4].\n"
        "- Packaging: The provided documents do not describe the packaging "
        "implementation, workflows, or encryption details."
    )
    generator = MockGenerator(default_response=grounded_answer)
    service = RagService(retriever=retriever, generator=generator)

    result = service.answer_question(query, top_k=2)

    assert result.has_sufficient_evidence is True
    assert result.cited_files == ["netflix_pipeline.pdf"]
    assert result.cited_pages == [
        ("netflix_pipeline.pdf", 3),
        ("netflix_pipeline.pdf", 4),
    ]
    assert result.directly_cited_indices == [1, 2]
    assert len(result.directly_cited_documents) == 2
    assert [c.document_index for c in result.directly_cited_documents] == [1, 2]
    assert len(result.citations) == 2

    cit1 = result.citations[0]
    assert cit1.document_index == 1
    assert cit1.file_name == "netflix_pipeline.pdf"
    assert cit1.page_number == 3
    assert "Ingestion validates" in cit1.snippet

    cit2 = result.citations[1]
    assert cit2.document_index == 2
    assert cit2.file_name == "netflix_pipeline.pdf"
    assert cit2.page_number == 4
    assert "Transcoding converts" in cit2.snippet

    # Verify that generator was passed prompt instructing distinction between
    # stated facts and missing info
    assert len(generator.calls) == 1
    prompt = generator.calls[0]
    assert "Distinguish explicitly stated facts from missing information" in prompt
    assert "Do NOT extrapolate, speculate, assume" in prompt


def test_rag_unsupported_details_safeguard() -> None:
    """Verify safeguard detects insufficient evidence for unsupported questions."""
    chunk_text = "The system uses Python 3.12 and FastAPI for web endpoints."
    embedder = MockEmbeddingModel(dimension=2, vectors={chunk_text: [1.0, 0.0]})
    retriever = SemanticRetriever(embedding_model=embedder)
    chunk = Chunk(
        text=chunk_text,
        metadata=ChunkMetadata(
            source="/docs/stack.txt",
            file_name="stack.txt",
            file_type="txt",
            page_number=None,
            chunk_index=0,
            start_char=0,
            end_char=len(chunk_text),
        ),
    )
    retriever.index.add(chunk, [1.0, 0.0])

    unsupported_query = (
        "What AES DRM encryption algorithms and license servers are used?"
    )
    retriever.embedding_model = MockEmbeddingModel(
        dimension=2, vectors={unsupported_query: [1.0, 0.0]}
    )

    insufficient_response = (
        "I do not have sufficient information in the provided documents to "
        "answer this question. The provided documents mention Python and FastAPI "
        "but do not describe AES DRM encryption algorithms or license servers."
    )
    generator = MockGenerator(default_response=insufficient_response)
    service = RagService(retriever=retriever, generator=generator)

    result = service.answer_question(unsupported_query, top_k=1)

    assert result.has_sufficient_evidence is False
    assert result.directly_cited_documents == []
    assert "do not have sufficient information" in result.answer.lower()


def test_rag_multi_part_question_cross_document_citations() -> None:
    """Verify multi-part question spanning multiple documents:

    partially supported parts are accurately cited to their respective files and pages,
    and unsupported parts are acknowledged without fabrication.
    """
    doc1_text = "Primary raw video assets are stored in Amazon S3 buckets."
    doc2_text = "Transcoding workers are orchestrated using AWS Batch worker pools."
    doc3_text = "Internal logging exports operational metrics to CloudWatch."

    vectors = {
        doc1_text: [1.0, 0.0, 0.0],
        doc2_text: [0.0, 1.0, 0.0],
        doc3_text: [0.0, 0.0, 1.0],
    }
    embedder = MockEmbeddingModel(dimension=3, vectors=vectors)
    index = VectorIndex(dimension=3)
    retriever = SemanticRetriever(embedding_model=embedder, index=index)

    c1 = Chunk(
        text=doc1_text,
        metadata=ChunkMetadata(
            source="/docs/storage_spec.pdf",
            file_name="storage_spec.pdf",
            file_type="pdf",
            page_number=2,
            chunk_index=0,
            start_char=0,
            end_char=len(doc1_text),
        ),
    )
    c2 = Chunk(
        text=doc2_text,
        metadata=ChunkMetadata(
            source="/docs/transcoder_spec.pdf",
            file_name="transcoder_spec.pdf",
            file_type="pdf",
            page_number=7,
            chunk_index=0,
            start_char=0,
            end_char=len(doc2_text),
        ),
    )
    c3 = Chunk(
        text=doc3_text,
        metadata=ChunkMetadata(
            source="/docs/telemetry.txt",
            file_name="telemetry.txt",
            file_type="txt",
            page_number=None,
            chunk_index=0,
            start_char=0,
            end_char=len(doc3_text),
        ),
    )
    retriever.index_chunks([c1, c2, c3])

    multi_query = (
        "1. What is the storage backend? 2. How are transcoders orchestrated? "
        "3. What is the customer subscription pricing?"
    )
    retriever.embedding_model = MockEmbeddingModel(
        dimension=3, vectors={multi_query: [0.8, 0.6, 0.2]}
    )

    multi_answer = (
        "1. Storage backend: Primary raw video assets are stored in Amazon S3 buckets "
        "[Document 1, Page 2].\n"
        "2. Transcoder orchestration: Transcoding workers are orchestrated using "
        "AWS Batch worker pools [Document 2, Page 7].\n"
        "3. Subscription pricing: The provided documents do not contain information "
        "regarding customer subscription pricing."
    )
    generator = MockGenerator(default_response=multi_answer)
    service = RagService(retriever=retriever, generator=generator)

    result = service.answer_question(multi_query, top_k=3)

    assert result.has_sufficient_evidence is True
    # Document 1 and 2 directly cited; Document 3 retrieved but not cited
    assert result.directly_cited_indices == [1, 2]
    assert len(result.directly_cited_documents) == 2
    assert [c.document_index for c in result.directly_cited_documents] == [1, 2]
    assert result.cited_files == [
        "storage_spec.pdf",
        "transcoder_spec.pdf",
        "telemetry.txt",
    ]
    assert result.cited_pages == [
        ("storage_spec.pdf", 2),
        ("transcoder_spec.pdf", 7),
        ("telemetry.txt", None),
    ]

    # Verify each citation's metadata
    assert len(result.citations) == 3
    assert result.citations[0].document_index == 1
    assert result.citations[0].file_name == "storage_spec.pdf"
    assert result.citations[0].page_number == 2

    assert result.citations[1].document_index == 2
    assert result.citations[1].file_name == "transcoder_spec.pdf"
    assert result.citations[1].page_number == 7

    assert result.citations[2].document_index == 3
    assert result.citations[2].file_name == "telemetry.txt"
    assert result.citations[2].page_number is None


def test_citation_correctness_and_snippet_formatting() -> None:
    """Verify precision of Citation construction and regex extraction."""
    long_text = "A" * 350
    chunk = Chunk(
        text=long_text,
        metadata=ChunkMetadata(
            source="/docs/large.pdf",
            file_name="large.pdf",
            file_type="pdf",
            page_number=15,
            chunk_index=2,
            start_char=0,
            end_char=350,
        ),
    )
    search_result = SearchResult(chunk=chunk, score=0.95)

    embedder = MockEmbeddingModel(dimension=2)
    retriever = SemanticRetriever(embedding_model=embedder)
    retriever.index.add(chunk, [1.0, 0.0])

    generator = MockGenerator(
        default_response="According to [Document 1, Page 15], this is true."
    )
    service = RagService(retriever=retriever, generator=generator)

    result = service.answer_question("Query", top_k=1)
    assert len(result.citations) == 1
    cit = result.citations[0]
    assert cit.document_index == 1
    assert cit.file_name == "large.pdf"
    assert cit.page_number == 15
    assert len(cit.snippet) <= 205
    assert cit.snippet.endswith("...")
    assert result.directly_cited_indices == [1]
    assert len(result.directly_cited_documents) == 1
    assert result.directly_cited_documents[0] == cit

    # Test edge case: answer with no citations returns empty directly_cited_documents
    no_cit_answer = RagAnswer(
        query="Q",
        answer="General summary with no brackets.",
        citations=[cit],
        evidence=[search_result],
    )
    assert no_cit_answer.directly_cited_documents == []


def test_rag_answer_preserves_provider_and_model() -> None:
    """Verify RagService populates provider and model on RagAnswer."""
    embedder = MockEmbeddingModel(dimension=2)
    retriever = SemanticRetriever(embedding_model=embedder)
    chunk = Chunk(
        text="Sample text for provider test.",
        metadata=ChunkMetadata(
            source="/docs/p.txt",
            file_name="p.txt",
            file_type="txt",
            page_number=1,
            chunk_index=0,
            start_char=0,
            end_char=30,
        ),
    )
    retriever.index.add(chunk, [1.0, 0.0])

    # Test with MockGenerator
    mock_gen = MockGenerator(default_response="Mock answer [Document 1, Page 1].")
    service_mock = RagService(retriever=retriever, generator=mock_gen)
    ans_mock = service_mock.answer_question("Question")
    assert ans_mock.provider == "mock"
    assert ans_mock.model == "mock"

    # Test with OllamaGenerator
    ollama_gen = OllamaGenerator(
        base_url="http://localhost:11434",
        model="llama3.2:3b",
    )
    from unittest.mock import patch

    with patch.object(
        ollama_gen,
        "generate",
        return_value="Grounded Ollama answer [Document 1, Page 1].",
    ):
        service_ollama = RagService(retriever=retriever, generator=ollama_gen)
        ans_ollama = service_ollama.answer_question("Question")
        assert ans_ollama.provider == "ollama"
        assert ans_ollama.model == "llama3.2:3b"
