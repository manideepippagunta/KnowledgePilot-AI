"""End-to-end integration tests for KnowledgePilot AI (Phase 5)."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from pypdf import PdfReader, PdfWriter

from knowledgepilot.app import main
from knowledgepilot.embeddings.mock import MockEmbeddingModel
from knowledgepilot.embeddings.sentence_transformer import (
    SentenceTransformerEmbedding,
)
from knowledgepilot.generation.mock import MockGenerator
from knowledgepilot.rag import ask_document


def _make_clean_page_pdf(text: str) -> bytes:
    """Construct a clean, valid single-page PDF containing the given text."""
    obj1 = b"1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n"
    obj2 = b"2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n"
    obj3 = (
        b"3 0 obj\n"
        b"<</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources <</Font <</F1 4 0 R>>>> /Contents 5 0 R>>\n"
        b"endobj\n"
    )
    obj4 = b"4 0 obj\n<</Type /Font /Subtype /Type1 /BaseFont /Helvetica>>\nendobj\n"
    stream = f"BT\n/F1 12 Tf\n72 712 Td\n({text}) Tj\nET\n".encode("latin-1")
    obj5 = (
        f"5 0 obj\n<</Length {len(stream)}>>\nstream\n".encode("latin-1")
        + stream
        + b"endstream\nendobj\n"
    )

    header = b"%PDF-1.4\n"
    off1 = len(header)
    off2 = off1 + len(obj1)
    off3 = off2 + len(obj2)
    off4 = off3 + len(obj3)
    off5 = off4 + len(obj4)
    xref_offset = off5 + len(obj5)

    xref = (
        b"xref\n0 6\n"
        b"0000000000 65535 f \n"
        + f"{off1:010d} 00000 n \n".encode()
        + f"{off2:010d} 00000 n \n".encode()
        + f"{off3:010d} 00000 n \n".encode()
        + f"{off4:010d} 00000 n \n".encode()
        + f"{off5:010d} 00000 n \n".encode()
    )
    trailer = (
        f"trailer\n<</Size 6 /Root 1 0 R>>\nstartxref\n{xref_offset}\n%%EOF\n".encode()
    )
    return header + obj1 + obj2 + obj3 + obj4 + obj5 + xref + trailer


def _create_pdf_file(path: Path, pages: list[str]) -> None:
    """Create a multi-page valid PDF on disk with the given text on each page."""
    writer = PdfWriter()
    for text in pages:
        reader = PdfReader(io.BytesIO(_make_clean_page_pdf(text)))
        writer.add_page(reader.pages[0])
    with open(path, "wb") as f:
        writer.write(f)


# ---------------------------------------------------------------------------
# Offline Deterministic Integration Tests (Standard Test Suite)
# ---------------------------------------------------------------------------


def test_offline_end_to_end_txt_integration(tmp_path: Path) -> None:
    doc_file = tmp_path / "overview.txt"
    doc_file.write_text(
        "KnowledgePilot AI provides document ingestion, vector retrieval, "
        "and grounded question answering.",
        encoding="utf-8",
    )

    embedder = MockEmbeddingModel(dimension=4)
    generator = MockGenerator(
        default_response="KnowledgePilot provides grounded question answering."
    )

    result = ask_document(
        doc_path=doc_file,
        query="What does KnowledgePilot provide?",
        embedding_model=embedder,
        generator=generator,
    )

    assert result.has_sufficient_evidence is True
    assert result.answer == "KnowledgePilot provides grounded question answering."
    assert result.cited_files == ["overview.txt"]
    assert result.cited_pages == [("overview.txt", 1)]
    assert len(result.evidence) >= 1
    assert "document ingestion" in result.evidence[0].text


def test_offline_end_to_end_pdf_with_page_citations(tmp_path: Path) -> None:
    pdf_file = tmp_path / "manual.pdf"
    p1 = "Introduction to document ingestion and chunking strategies."
    p2 = "Cosine similarity equations and high-dimensional vector search."
    _create_pdf_file(pdf_file, [p1, p2])

    vectors = {
        p1: [1.0, 0.0],
        p2: [0.0, 1.0],
        "cosine similarity question": [0.0, 1.0],
    }
    embedder = MockEmbeddingModel(dimension=2, vectors=vectors)
    generator = MockGenerator(
        default_response="Cosine similarity measures the angle between vectors."
    )

    result = ask_document(
        doc_path=pdf_file,
        query="cosine similarity question",
        embedding_model=embedder,
        generator=generator,
    )

    assert result.has_sufficient_evidence is True
    assert len(result.citations) >= 1
    assert result.cited_files == ["manual.pdf"]
    # Evidence must attribute specifically to Page 2
    assert result.citations[0].page_number == 2
    assert result.citations[0].file_name == "manual.pdf"
    assert "Cosine similarity equations" in result.evidence[0].text


def test_cli_doc_query_with_mock_embeddings(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    doc_file = tmp_path / "guide.txt"
    doc_file.write_text(
        "KnowledgePilot CLI supports question answering directly from files.",
        encoding="utf-8",
    )

    code = main(
        [
            "--doc",
            str(doc_file),
            "--query",
            "What does the CLI support?",
            "--mock-embeddings",
        ]
    )

    assert code == 0
    out = capsys.readouterr().out
    assert "KnowledgePilot AI - Grounded RAG Answer" in out
    assert "guide.txt" in out
    assert "Citations" in out
    assert "MockGenerator" in out
    assert "Evidence Chunks" in out


def test_cli_missing_query_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    doc_file = tmp_path / "doc.txt"
    doc_file.write_text("Some text.", encoding="utf-8")

    code = main(["--doc", str(doc_file)])
    assert code == 2
    err = capsys.readouterr().err
    assert "Error: --query (-q) is required when --doc is specified." in err


def test_cli_doc_file_not_found(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["--doc", "non_existent_file.pdf", "--query", "What is this?"])
    assert code == 2
    err = capsys.readouterr().err
    assert "Error: Document file not found" in err


# ---------------------------------------------------------------------------
# Real Model Integration Test (Explicitly run with: pytest -m integration)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_real_sentence_transformer_integration(tmp_path: Path) -> None:
    """Integration test using the real sentence-transformer model.

    Validates that real dense embeddings correctly rank semantically relevant
    PDF pages and pass evidence with accurate page citations to the RAG layer.
    """
    pdf_file = tmp_path / "deep_learning.pdf"
    page1 = (
        "Machine learning neural networks optimize parameter weights "
        "via backpropagation and gradient descent algorithms."
    )
    page2 = (
        "Relational database management systems execute structured SQL queries "
        "and maintain ACID transactional guarantees."
    )
    _create_pdf_file(pdf_file, [page1, page2])

    real_embedder = SentenceTransformerEmbedding(model_name="all-MiniLM-L6-v2")
    generator = MockGenerator(
        default_response="Neural networks use gradient descent to optimize parameters."
    )

    result = ask_document(
        doc_path=pdf_file,
        query="How do neural networks optimize their weights?",
        top_k=1,
        embedding_model=real_embedder,
        generator=generator,
    )

    assert result.has_sufficient_evidence is True
    assert len(result.citations) == 1
    top_citation = result.citations[0]

    # Real semantic search must rank Page 1 first
    assert top_citation.page_number == 1
    assert top_citation.file_name == "deep_learning.pdf"
    assert "neural networks" in result.evidence[0].text.lower()
    assert result.evidence[0].score > 0.4
