"""Unit tests for knowledgepilot.ingestion (Phase 2)."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from pypdf import PdfReader, PdfWriter

from knowledgepilot.ingestion import (
    Chunk,
    ChunkMetadata,
    Document,
    DocumentMetadata,
    EmptyDocumentError,
    MalformedDocumentError,
    PdfLoader,
    TextChunker,
    TxtLoader,
    UnsupportedFileTypeError,
    load_document,
)

# ---------------------------------------------------------------------------
# Test Helpers for PDF generation
# ---------------------------------------------------------------------------


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


def _create_blank_pdf_file(path: Path, page_count: int = 1) -> None:
    """Create a PDF with blank pages (no text content)."""
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=100, height=100)
    with open(path, "wb") as f:
        writer.write(f)


def _create_encrypted_pdf_file(path: Path, password: str = "secret") -> None:
    """Create a password-encrypted PDF on disk."""
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.encrypt(password)
    with open(path, "wb") as f:
        writer.write(f)


# ---------------------------------------------------------------------------
# Models & Immutability Tests
# ---------------------------------------------------------------------------


def test_document_and_metadata_frozen() -> None:
    meta = DocumentMetadata(
        source="/path/to/doc.txt",
        file_name="doc.txt",
        file_type="txt",
        page_number=1,
        total_pages=1,
        extra={"author": "Alice"},
    )
    doc = Document(text="Sample content", metadata=meta)

    assert doc.text == "Sample content"
    assert doc.metadata.source == "/path/to/doc.txt"
    assert doc.metadata.file_name == "doc.txt"
    assert doc.metadata.file_type == "txt"
    assert doc.metadata.page_number == 1
    assert doc.metadata.total_pages == 1
    assert doc.metadata.extra == {"author": "Alice"}

    with pytest.raises(AttributeError):
        doc.text = "Modified"  # type: ignore[misc]

    with pytest.raises(AttributeError):
        meta.file_name = "other.txt"  # type: ignore[misc]


def test_chunk_and_metadata_frozen() -> None:
    meta = ChunkMetadata(
        source="/path/to/doc.txt",
        file_name="doc.txt",
        file_type="txt",
        page_number=1,
        chunk_index=0,
        start_char=0,
        end_char=10,
        extra={"key": "val"},
    )
    chunk = Chunk(text="Sample ten", metadata=meta)

    assert chunk.text == "Sample ten"
    assert chunk.metadata.chunk_index == 0
    assert chunk.metadata.start_char == 0
    assert chunk.metadata.end_char == 10
    assert chunk.metadata.extra == {"key": "val"}

    with pytest.raises(AttributeError):
        chunk.text = "Changed"  # type: ignore[misc]

    with pytest.raises(AttributeError):
        meta.chunk_index = 5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TXT Ingestion Tests
# ---------------------------------------------------------------------------


def test_txt_loader_valid_document(tmp_path: Path) -> None:
    file = tmp_path / "sample.txt"
    file.write_text("KnowledgePilot AI is a production RAG system.", encoding="utf-8")

    docs = TxtLoader().load(file)
    assert len(docs) == 1
    doc = docs[0]
    assert doc.text == "KnowledgePilot AI is a production RAG system."
    assert doc.metadata.file_name == "sample.txt"
    assert doc.metadata.file_type == "txt"
    assert doc.metadata.page_number == 1
    assert doc.metadata.total_pages == 1
    assert doc.metadata.source == str(file.resolve())


def test_txt_loader_empty_raises(tmp_path: Path) -> None:
    file = tmp_path / "empty.txt"
    file.write_text("   \n\t  ", encoding="utf-8")

    with pytest.raises(EmptyDocumentError, match="contains no extractable text"):
        TxtLoader(allow_empty=False).load(file)


def test_txt_loader_empty_allowed(tmp_path: Path) -> None:
    file = tmp_path / "empty.txt"
    file.write_text("", encoding="utf-8")

    docs = TxtLoader(allow_empty=True).load(file)
    assert docs == []


def test_txt_loader_nonexistent_raises(tmp_path: Path) -> None:
    file = tmp_path / "missing.txt"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        TxtLoader().load(file)


def test_txt_loader_directory_raises(tmp_path: Path) -> None:
    dir_path = tmp_path / "subfolder"
    dir_path.mkdir()
    with pytest.raises(ValueError, match="not a regular file"):
        TxtLoader().load(dir_path)


def test_txt_loader_malformed_encoding(tmp_path: Path) -> None:
    file = tmp_path / "bad_bytes.txt"
    # Write invalid UTF-8 sequence
    file.write_bytes(b"\x80\x81\xff\xfe")

    with pytest.raises(MalformedDocumentError, match="Failed to decode text file"):
        TxtLoader().load(file)


# ---------------------------------------------------------------------------
# PDF Ingestion Tests
# ---------------------------------------------------------------------------


def test_pdf_loader_single_page(tmp_path: Path) -> None:
    file = tmp_path / "single.pdf"
    _create_pdf_file(file, ["Page 1 text content"])

    docs = PdfLoader().load(file)
    assert len(docs) == 1
    doc = docs[0]
    assert doc.text == "Page 1 text content"
    assert doc.metadata.file_name == "single.pdf"
    assert doc.metadata.file_type == "pdf"
    assert doc.metadata.page_number == 1
    assert doc.metadata.total_pages == 1
    assert doc.metadata.source == str(file.resolve())


def test_pdf_loader_multi_page(tmp_path: Path) -> None:
    file = tmp_path / "multi.pdf"
    _create_pdf_file(file, ["Introduction to RAG", "Advanced Vector Retrieval"])

    docs = PdfLoader().load(file)
    assert len(docs) == 2

    assert docs[0].text == "Introduction to RAG"
    assert docs[0].metadata.page_number == 1
    assert docs[0].metadata.total_pages == 2
    assert docs[0].metadata.file_type == "pdf"

    assert docs[1].text == "Advanced Vector Retrieval"
    assert docs[1].metadata.page_number == 2
    assert docs[1].metadata.total_pages == 2
    assert docs[1].metadata.file_type == "pdf"


def test_pdf_loader_empty_file_raises(tmp_path: Path) -> None:
    file = tmp_path / "zero_bytes.pdf"
    file.write_bytes(b"")

    with pytest.raises(EmptyDocumentError, match="empty"):
        PdfLoader(allow_empty=False).load(file)


def test_pdf_loader_empty_file_allowed(tmp_path: Path) -> None:
    file = tmp_path / "zero_bytes.pdf"
    file.write_bytes(b"")

    docs = PdfLoader(allow_empty=True).load(file)
    assert docs == []


def test_pdf_loader_blank_pages_raises(tmp_path: Path) -> None:
    file = tmp_path / "blank.pdf"
    _create_blank_pdf_file(file, page_count=2)

    with pytest.raises(EmptyDocumentError, match="contains no extractable text"):
        PdfLoader(allow_empty=False).load(file)


def test_pdf_loader_blank_pages_allowed(tmp_path: Path) -> None:
    file = tmp_path / "blank.pdf"
    _create_blank_pdf_file(file, page_count=2)

    docs = PdfLoader(allow_empty=True).load(file)
    assert docs == []


def test_pdf_loader_malformed_corrupt(tmp_path: Path) -> None:
    file = tmp_path / "corrupted.pdf"
    file.write_bytes(b"%PDF-corrupted-random-bytes-123456789")

    with pytest.raises(MalformedDocumentError, match="Malformed or corrupted PDF"):
        PdfLoader().load(file)


def test_pdf_loader_encrypted_raises(tmp_path: Path) -> None:
    file = tmp_path / "encrypted.pdf"
    _create_encrypted_pdf_file(file, password="secret_password")

    with pytest.raises(MalformedDocumentError, match="password protected or encrypted"):
        PdfLoader().load(file)


def test_pdf_loader_directory_raises(tmp_path: Path) -> None:
    dir_path = tmp_path / "pdf_folder"
    dir_path.mkdir()
    with pytest.raises(ValueError, match="not a regular file"):
        PdfLoader().load(dir_path)


# ---------------------------------------------------------------------------
# load_document Dispatcher Tests
# ---------------------------------------------------------------------------


def test_load_document_txt(tmp_path: Path) -> None:
    file = tmp_path / "doc.txt"
    file.write_text("Hello from txt", encoding="utf-8")

    docs = load_document(file)
    assert len(docs) == 1
    assert docs[0].text == "Hello from txt"
    assert docs[0].metadata.file_type == "txt"


def test_load_document_pdf(tmp_path: Path) -> None:
    file = tmp_path / "doc.PDF"  # test uppercase extension
    _create_pdf_file(file, ["Hello from PDF"])

    docs = load_document(file)
    assert len(docs) == 1
    assert docs[0].text == "Hello from PDF"
    assert docs[0].metadata.file_type == "pdf"


def test_load_document_unsupported_extensions(tmp_path: Path) -> None:
    for ext in [".docx", ".csv", ".json", ".md", ".html", ""]:
        file = tmp_path / f"test{ext}"
        file.write_text("content", encoding="utf-8")
        with pytest.raises(UnsupportedFileTypeError, match="Unsupported file format"):
            load_document(file)


def test_load_document_nonexistent_raises(tmp_path: Path) -> None:
    file = tmp_path / "does_not_exist.pdf"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_document(file)


# ---------------------------------------------------------------------------
# TextChunker & Chunk Boundaries Tests
# ---------------------------------------------------------------------------


def test_chunker_validation_chunk_size() -> None:
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        TextChunker(chunk_size=0, chunk_overlap=0)

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        TextChunker(chunk_size=-10, chunk_overlap=0)


def test_chunker_validation_chunk_overlap() -> None:
    with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
        TextChunker(chunk_size=100, chunk_overlap=-1)

    with pytest.raises(ValueError, match="strictly less than chunk_size"):
        TextChunker(chunk_size=100, chunk_overlap=100)

    with pytest.raises(ValueError, match="strictly less than chunk_size"):
        TextChunker(chunk_size=100, chunk_overlap=150)


def test_chunker_empty_and_whitespace() -> None:
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    assert chunker.split_text("") == []
    assert chunker.split_text("   \n\t  ") == []


def test_chunker_shorter_than_chunk_size() -> None:
    chunker = TextChunker(chunk_size=50, chunk_overlap=10)
    text = "Short text."
    chunks = chunker.split_text(text)
    assert len(chunks) == 1
    chunk_text, start, end = chunks[0]
    assert chunk_text == text
    assert start == 0
    assert end == len(text)


def test_chunker_exact_chunk_size() -> None:
    chunker = TextChunker(chunk_size=10, chunk_overlap=2)
    text = "0123456789"
    chunks = chunker.split_text(text)
    assert len(chunks) == 1
    assert chunks[0] == ("0123456789", 0, 10)


def test_chunker_chunk_size_plus_one_boundary() -> None:
    # chunk_size=10, overlap=3, step=7
    # text len=11 -> "0123456789a"
    # Chunk 0: text[0:10] = "0123456789", start=0, end=10
    # Chunk 1: text[7:11] = "789a", start=7, end=11
    # Overlap between Chunk 0 and Chunk 1: "789" (len=3)
    chunker = TextChunker(chunk_size=10, chunk_overlap=3)
    text = "0123456789a"
    chunks = chunker.split_text(text)
    assert len(chunks) == 2

    assert chunks[0] == ("0123456789", 0, 10)
    assert chunks[1] == ("789a", 7, 11)

    # Verify overlap
    assert chunks[0][0][-3:] == chunks[1][0][:3]


def test_chunker_zero_overlap() -> None:
    chunker = TextChunker(chunk_size=5, chunk_overlap=0)
    text = "abcdefghijklm"  # len=13
    chunks = chunker.split_text(text)
    assert len(chunks) == 3
    assert chunks[0] == ("abcde", 0, 5)
    assert chunks[1] == ("fghij", 5, 10)
    assert chunks[2] == ("klm", 10, 13)


def test_split_document_metadata_propagation() -> None:
    chunker = TextChunker(chunk_size=10, chunk_overlap=2)
    meta = DocumentMetadata(
        source="/path/to/manual.pdf",
        file_name="manual.pdf",
        file_type="pdf",
        page_number=3,
        total_pages=10,
        extra={"section": "intro"},
    )
    doc = Document(text="0123456789abcdef", metadata=meta)

    chunks = chunker.split_document(doc, start_chunk_index=100)
    assert len(chunks) == 2

    c0 = chunks[0]
    assert c0.text == "0123456789"
    assert c0.metadata.source == "/path/to/manual.pdf"
    assert c0.metadata.file_name == "manual.pdf"
    assert c0.metadata.file_type == "pdf"
    assert c0.metadata.page_number == 3
    assert c0.metadata.chunk_index == 100
    assert c0.metadata.start_char == 0
    assert c0.metadata.end_char == 10
    assert c0.metadata.extra == {"section": "intro"}

    c1 = chunks[1]
    assert c1.text == "89abcdef"
    assert c1.metadata.page_number == 3
    assert c1.metadata.chunk_index == 101
    assert c1.metadata.start_char == 8
    assert c1.metadata.end_char == 16


def test_split_documents_multi_page_pdf_continuity() -> None:
    chunker = TextChunker(chunk_size=15, chunk_overlap=5)
    doc_page_1 = Document(
        text="Page one first sentence. Page one second sentence.",
        metadata=DocumentMetadata(
            source="/docs/whitepaper.pdf",
            file_name="whitepaper.pdf",
            file_type="pdf",
            page_number=1,
            total_pages=2,
        ),
    )
    doc_page_2 = Document(
        text="Page two starts here. Page two concludes.",
        metadata=DocumentMetadata(
            source="/docs/whitepaper.pdf",
            file_name="whitepaper.pdf",
            file_type="pdf",
            page_number=2,
            total_pages=2,
        ),
    )

    chunks = chunker.split_documents([doc_page_1, doc_page_2])
    assert len(chunks) > 2

    # Check monotonic chunk indexing
    for i, chunk in enumerate(chunks):
        assert chunk.metadata.chunk_index == i

    # Page 1 chunks must have page_number=1
    page_1_chunks = [c for c in chunks if c.metadata.page_number == 1]
    assert len(page_1_chunks) >= 1
    for c in page_1_chunks:
        assert c.metadata.page_number == 1
        assert c.metadata.file_name == "whitepaper.pdf"

    # Page 2 chunks must have page_number=2
    page_2_chunks = [c for c in chunks if c.metadata.page_number == 2]
    assert len(page_2_chunks) >= 1
    for c in page_2_chunks:
        assert c.metadata.page_number == 2
        assert c.metadata.file_name == "whitepaper.pdf"
