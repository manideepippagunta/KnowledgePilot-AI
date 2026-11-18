"""Tests for KnowledgePilot AI Web API and endpoints (Phase 7)."""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from knowledgepilot.config import Settings
from knowledgepilot.embeddings.mock import MockEmbeddingModel
from knowledgepilot.generation.mock import MockGenerator
from knowledgepilot.generation.ollama import OllamaGenerator
from knowledgepilot.web.app import create_web_app
from knowledgepilot.web.state import (
    KnowledgeStore,
    compute_sha256,
    sanitize_filename,
)


def _make_minimal_pdf_bytes(text: str = "Test PDF document content.") -> bytes:
    """Create a minimal valid PDF byte sequence for upload testing."""
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


class WebTestClient:
    """Synchronous test client wrapper around httpx.AsyncClient with ASGITransport."""

    def __init__(self, app: FastAPI, base_url: str = "http://testserver") -> None:
        self.app = app
        self.base_url = base_url
        self.transport = ASGITransport(app=app)

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        async def _call() -> httpx.Response:
            async with AsyncClient(
                transport=self.transport, base_url=self.base_url
            ) as client:
                return await client.request(method, url, **kwargs)

        return asyncio.run(_call())

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("POST", url, **kwargs)

    def delete(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("DELETE", url, **kwargs)


@pytest.fixture
def test_client() -> WebTestClient:
    """Create a WebTestClient with mock embeddings and deterministic mock generator."""
    store = KnowledgeStore(
        embedding_model=MockEmbeddingModel(dimension=4),
        generator=MockGenerator(
            default_response="KnowledgePilot AI is a production RAG application."
        ),
    )
    settings = Settings(env="test", log_level="INFO")
    app = create_web_app(settings=settings, store=store)
    return WebTestClient(app)


def test_status_endpoint(test_client: WebTestClient) -> None:
    resp = test_client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["provider"] == "mock"
    assert data["total_documents"] == 0
    assert data["total_chunks"] == 0
    assert data["ollama_available"] is False


def test_status_with_ollama_generator() -> None:
    gen = OllamaGenerator()
    store = KnowledgeStore(
        embedding_model=MockEmbeddingModel(dimension=4),
        generator=gen,
    )
    app = create_web_app(store=store)
    client = WebTestClient(app)

    with patch.object(gen, "is_available", return_value=True):
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["provider"] == "ollama"
        assert data["ollama_available"] is True


def test_upload_txt_document(test_client: WebTestClient) -> None:
    content = b"KnowledgePilot AI provides grounded document question answering."
    files = {"file": ("guide.txt", io.BytesIO(content), "text/plain")}

    resp = test_client.post("/api/documents/upload", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["file_name"] == "guide.txt"
    assert data["file_type"] == "txt"
    assert data["chunk_count"] >= 1
    assert data["page_count"] == 1

    # Verify status reflects indexed document
    status_resp = test_client.get("/api/status")
    assert status_resp.json()["total_documents"] == 1
    assert status_resp.json()["total_chunks"] >= 1


def test_upload_pdf_document(test_client: WebTestClient) -> None:
    pdf_bytes = _make_minimal_pdf_bytes("PDF documentation on vector retrieval.")
    files = {"file": ("manual.pdf", io.BytesIO(pdf_bytes), "application/pdf")}

    resp = test_client.post("/api/documents/upload", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["file_name"] == "manual.pdf"
    assert data["file_type"] == "pdf"
    assert data["page_count"] == 1
    assert data["chunk_count"] >= 1


def test_upload_unsupported_file_type(test_client: WebTestClient) -> None:
    files = {"file": ("data.csv", io.BytesIO(b"a,b,c\n1,2,3"), "text/csv")}
    resp = test_client.post("/api/documents/upload", files=files)
    assert resp.status_code == 400
    assert "Unsupported file format" in resp.json()["detail"]


def test_upload_empty_file(test_client: WebTestClient) -> None:
    files = {"file": ("empty.txt", io.BytesIO(b""), "text/plain")}
    resp = test_client.post("/api/documents/upload", files=files)
    assert resp.status_code == 400
    assert "is empty" in resp.json()["detail"]


def test_list_and_clear_documents(test_client: WebTestClient) -> None:
    # Upload one document
    test_client.post(
        "/api/documents/upload",
        files={"file": ("doc.txt", io.BytesIO(b"Some text."), "text/plain")},
    )

    list_resp = test_client.get("/api/documents")
    assert list_resp.status_code == 200
    assert len(list_resp.json()) == 1
    assert list_resp.json()[0]["file_name"] == "doc.txt"

    # Clear documents
    del_resp = test_client.delete("/api/documents")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "cleared"

    # Verify empty list
    list_after = test_client.get("/api/documents")
    assert list_after.json() == []


def test_chat_grounded_answer_with_citations(test_client: WebTestClient) -> None:
    # Upload document
    text = (
        "KnowledgePilot AI indexes documents into vector space and "
        "generates verified answers using retrieved evidence."
    )
    test_client.post(
        "/api/documents/upload",
        files={"file": ("architecture.txt", io.BytesIO(text.encode()), "text/plain")},
    )

    # Ask question
    chat_resp = test_client.post(
        "/api/chat",
        json={"query": "What does KnowledgePilot AI do?", "top_k": 2},
    )
    assert chat_resp.status_code == 200
    data = chat_resp.json()
    assert data["query"] == "What does KnowledgePilot AI do?"
    assert data["has_sufficient_evidence"] is True
    assert "KnowledgePilot AI" in data["answer"]
    assert len(data["citations"]) >= 1
    assert data["citations"][0]["document_index"] == 1
    assert data["citations"][0]["file_name"] == "architecture.txt"
    assert data["citations"][0]["page_number"] == 1
    assert "indexes documents" in data["citations"][0]["snippet"]
    assert len(data["evidence"]) >= 1
    assert data["evidence"][0]["document_index"] == 1
    assert data["evidence"][0]["score"] > 0.0


def test_chat_insufficient_evidence_when_store_empty(
    test_client: WebTestClient,
) -> None:
    resp = test_client.post(
        "/api/chat",
        json={"query": "What is the meaning of life?", "top_k": 3},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["has_sufficient_evidence"] is False
    assert (
        "do not have sufficient information" in data["answer"].lower()
        or "insufficient" in data["answer"].lower()
    )
    assert data["citations"] == []
    assert data["evidence"] == []


def test_chat_validation_error_on_empty_query(test_client: WebTestClient) -> None:
    resp = test_client.post("/api/chat", json={"query": ""})
    assert resp.status_code == 422


def test_chat_with_ollama_override(test_client: WebTestClient) -> None:
    test_client.post(
        "/api/documents/upload",
        files={"file": ("doc.txt", io.BytesIO(b"System info."), "text/plain")},
    )

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = (
        b'{"response": "Ollama grounded answer.", "done": true}'
    )
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None

    with patch("urllib.request.urlopen", return_value=mock_resp):
        resp = test_client.post(
            "/api/chat",
            json={
                "query": "Tell me about the system.",
                "generator": "ollama",
                "model": "llama3.2:1b",
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["provider"] == "ollama"
    assert data["model"] == "llama3.2:1b"
    assert data["answer"] == "Ollama grounded answer."


def test_serve_index_html(test_client: WebTestClient) -> None:
    resp = test_client.get("/")
    assert resp.status_code == 200
    assert "KnowledgePilot" in resp.text
    assert "style.css" in resp.text


def test_serve_static_css(test_client: WebTestClient) -> None:
    resp = test_client.get("/static/style.css")
    assert resp.status_code == 200
    assert "--bg-core" in resp.text


def test_sanitize_filename_edge_cases() -> None:
    assert sanitize_filename("") == "document"
    assert sanitize_filename("sample.txt") == "sample.txt"
    assert sanitize_filename("C:\\fakepath\\sample.txt") == "sample.txt"
    assert sanitize_filename("/var/uploads/report.pdf") == "report.pdf"
    assert sanitize_filename("sample:1.txt") == "sample_1.txt"
    assert sanitize_filename("test?.txt") == "test_.txt"
    assert sanitize_filename("file*name.txt") == "file_name.txt"
    assert sanitize_filename('<doc>|test".txt') == "_doc__test_.txt"
    assert sanitize_filename("...txt") == "document.txt"
    assert sanitize_filename(".txt") == "document.txt"
    assert sanitize_filename("  sample.txt  ") == "sample.txt"
    assert sanitize_filename("sample.txt. ") == "sample.txt"


def test_upload_txt_with_windows_invalid_filename(test_client: WebTestClient) -> None:
    content = b"Grounded KnowledgePilot AI document content with strange filename."
    files = {"file": ("test?.txt", io.BytesIO(content), "text/plain")}

    resp = test_client.post("/api/documents/upload", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["file_name"] == "test_.txt"
    assert data["chunk_count"] >= 1

    # Verify visible in /api/documents
    docs_resp = test_client.get("/api/documents")
    assert docs_resp.status_code == 200
    assert any(d["file_name"] == "test_.txt" for d in docs_resp.json())


def test_upload_with_windows_absolute_path_filename(
    test_client: WebTestClient,
) -> None:
    content = b"System architecture specification."
    files = {
        "file": (
            "C:\\fakepath\\system_spec.txt",
            io.BytesIO(content),
            "text/plain",
        )
    }

    resp = test_client.post("/api/documents/upload", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["file_name"] == "system_spec.txt"

    # Ask question and verify citation uses sanitized filename
    chat_resp = test_client.post(
        "/api/chat",
        json={"query": "What is the system architecture?", "top_k": 1},
    )
    assert chat_resp.status_code == 200
    chat_data = chat_resp.json()
    assert len(chat_data["citations"]) >= 1
    assert chat_data["citations"][0]["file_name"] == "system_spec.txt"
    assert chat_data["citations"][0]["source"] == "system_spec.txt"


def test_upload_pdf_preserves_page_metadata_in_citations(
    test_client: WebTestClient,
) -> None:
    pdf_bytes = _make_minimal_pdf_bytes("Deep semantic retrieval documentation page.")
    files = {"file": ("manual_guide.pdf", io.BytesIO(pdf_bytes), "application/pdf")}

    resp = test_client.post("/api/documents/upload", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["file_name"] == "manual_guide.pdf"
    assert data["page_count"] == 1

    chat_resp = test_client.post(
        "/api/chat",
        json={"query": "What is the documentation about?", "top_k": 1},
    )
    assert chat_resp.status_code == 200
    citations = chat_resp.json()["citations"]
    assert len(citations) >= 1
    assert citations[0]["file_name"] == "manual_guide.pdf"
    assert citations[0]["page_number"] == 1


def test_upload_duplicate_filenames(test_client: WebTestClient) -> None:
    # Upload first copy
    resp1 = test_client.post(
        "/api/documents/upload",
        files={
            "file": ("notes.txt", io.BytesIO(b"First version of notes."), "text/plain")
        },
    )
    assert resp1.status_code == 200
    doc1_id = resp1.json()["doc_id"]

    # Upload second copy with same filename
    resp2 = test_client.post(
        "/api/documents/upload",
        files={
            "file": (
                "notes.txt",
                io.BytesIO(b"Second version of notes."),
                "text/plain",
            )
        },
    )
    assert resp2.status_code == 200
    doc2_id = resp2.json()["doc_id"]

    # Verify both exist with distinct doc_ids
    assert doc1_id != doc2_id

    list_resp = test_client.get("/api/documents")
    assert list_resp.status_code == 200
    summaries = list_resp.json()
    assert len(summaries) == 2
    assert all(d["file_name"] == "notes.txt" for d in summaries)

    status_resp = test_client.get("/api/status")
    assert status_resp.json()["total_documents"] == 2
    assert status_resp.json()["total_chunks"] >= 2


def test_upload_malformed_file_leaves_index_uncorrupted(
    test_client: WebTestClient,
) -> None:
    # 1. Ingest a valid document first
    valid_resp = test_client.post(
        "/api/documents/upload",
        files={
            "file": (
                "valid.txt",
                io.BytesIO(b"Valid baseline content."),
                "text/plain",
            )
        },
    )
    assert valid_resp.status_code == 200
    initial_chunks = valid_resp.json()["chunk_count"]

    # 2. Upload a corrupted PDF
    bad_pdf = b"%PDF-corrupted-binary-garbage-123456789"
    bad_resp = test_client.post(
        "/api/documents/upload",
        files={"file": ("corrupt.pdf", io.BytesIO(bad_pdf), "application/pdf")},
    )
    assert bad_resp.status_code == 400

    # 3. Verify store and index remain completely uncorrupted
    status_resp = test_client.get("/api/status")
    data = status_resp.json()
    assert data["total_documents"] == 1
    assert data["total_chunks"] == initial_chunks

    list_resp = test_client.get("/api/documents")
    assert len(list_resp.json()) == 1
    assert list_resp.json()[0]["file_name"] == "valid.txt"


def test_upload_temporary_file_cleaned_up(test_client: WebTestClient) -> None:
    import tempfile
    from pathlib import Path

    created_dirs: list[str] = []
    orig_tempdir = tempfile.TemporaryDirectory

    def tracking_tempdir(*args: Any, **kwargs: Any) -> Any:
        td = orig_tempdir(*args, **kwargs)
        created_dirs.append(td.name)
        return td

    with patch("tempfile.TemporaryDirectory", side_effect=tracking_tempdir):
        resp = test_client.post(
            "/api/documents/upload",
            files={
                "file": (
                    "cleanup_test.txt",
                    io.BytesIO(b"Data to verify cleanup."),
                    "text/plain",
                )
            },
        )
        assert resp.status_code == 200

    assert len(created_dirs) >= 1
    # Verify temporary directories were cleaned up after upload completed
    for dir_path in created_dirs:
        assert not Path(dir_path).exists()


def test_generator_endpoint_selects_ollama(test_client: WebTestClient) -> None:
    """Verify POST /api/generator updates store generator to Ollama."""
    resp = test_client.post(
        "/api/generator",
        json={"provider": "ollama", "model": "llama3.2:3b"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["provider"] == "ollama"
    assert data["model"] == "llama3.2:3b"

    current_gen = test_client.app.state.store.generator
    assert isinstance(current_gen, OllamaGenerator)
    assert current_gen.model == "llama3.2:3b"


def test_generator_endpoint_selects_mock(test_client: WebTestClient) -> None:
    """Verify POST /api/generator switches store generator back to Mock."""
    test_client.post("/api/generator", json={"provider": "ollama"})
    resp = test_client.post("/api/generator", json={"provider": "mock"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["provider"] == "mock"

    current_gen = test_client.app.state.store.generator
    assert isinstance(current_gen, MockGenerator)


def test_generator_endpoint_rejects_invalid_provider(
    test_client: WebTestClient,
) -> None:
    """Verify POST /api/generator validates provider names."""
    resp = test_client.post(
        "/api/generator",
        json={"provider": "unsupported_llm"},
    )
    assert resp.status_code == 400
    assert "Unsupported generator provider" in resp.json()["detail"]


def test_chat_with_ollama_generator_selection_and_badges(
    test_client: WebTestClient,
) -> None:
    """Verify selecting Ollama in chat calls Ollama with llama3.2:3b."""
    test_client.post(
        "/api/documents/upload",
        files={
            "file": (
                "netflix.txt",
                io.BytesIO(b"Netflix pipeline uses ingestion and transcoding."),
                "text/plain",
            )
        },
    )

    expected_answer = (
        "Netflix uses ingestion and transcoding as documented in [Document 1]."
    )
    with patch.object(OllamaGenerator, "generate", return_value=expected_answer):
        resp = test_client.post(
            "/api/chat",
            json={
                "query": "What does Netflix pipeline use?",
                "generator": "ollama",
                "model": "llama3.2:3b",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["provider"] == "ollama"
        assert data["model"] == "llama3.2:3b"
        assert data["answer"] == expected_answer
        assert data["has_sufficient_evidence"] is True

        store_gen = test_client.app.state.store.generator
        assert isinstance(store_gen, OllamaGenerator)
        assert store_gen.model == "llama3.2:3b"


def test_chat_with_mock_generator_selection_and_badges(
    test_client: WebTestClient,
) -> None:
    """Verify switching back to mock in chat sets provider and model to mock."""
    test_client.post(
        "/api/documents/upload",
        files={
            "file": (
                "demo.txt",
                io.BytesIO(b"Demo content for mock generation."),
                "text/plain",
            )
        },
    )
    test_client.post("/api/generator", json={"provider": "ollama"})

    resp = test_client.post(
        "/api/chat",
        json={
            "query": "What is in demo?",
            "generator": "mock",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["provider"] == "mock"
    assert data["model"] == "mock"

    store_gen = test_client.app.state.store.generator
    assert isinstance(store_gen, MockGenerator)


def test_chat_invalid_generator_raises_400(test_client: WebTestClient) -> None:
    """Verify /api/chat rejects unsupported generator values."""
    resp = test_client.post(
        "/api/chat",
        json={"query": "Test", "generator": "nonexistent"},
    )
    assert resp.status_code == 400
    assert "Unsupported generator provider" in resp.json()["detail"]


def test_chat_stream_empty_docs_returns_insufficient(
    test_client: WebTestClient,
) -> None:
    """Verify /api/chat/stream without documents emits insufficient evidence."""
    resp = test_client.post(
        "/api/chat/stream",
        json={"query": "What is KnowledgePilot?"},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    text = resp.text
    assert "data: " in text
    # Parse data lines
    events = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))

    assert len(events) == 1
    done_evt = events[0]
    assert done_evt["type"] == "done"
    assert done_evt["has_sufficient_evidence"] is False
    assert done_evt["citations"] == []


def test_chat_stream_with_document_yields_tokens_and_done(
    test_client: WebTestClient,
) -> None:
    """Verify /api/chat/stream streams tokens and sends final consolidated event."""
    test_client.post(
        "/api/documents/upload",
        files={
            "file": (
                "overview.txt",
                io.BytesIO(b"KnowledgePilot AI provides real-time streaming RAG."),
                "text/plain",
            )
        },
    )

    resp = test_client.post(
        "/api/chat/stream",
        json={"query": "What does KnowledgePilot provide?", "generator": "mock"},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    events = []
    for line in resp.text.strip().split("\n"):
        line = line.strip()
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))

    token_events = [e for e in events if e.get("type") == "token"]
    done_events = [e for e in events if e.get("type") == "done"]

    assert len(token_events) > 0
    assert len(done_events) == 1

    done_evt = done_events[0]
    assert done_evt["has_sufficient_evidence"] is True
    assert len(done_evt["citations"]) >= 1
    assert done_evt["citations"][0]["file_name"] == "overview.txt"
    assert len(done_evt["evidence"]) >= 1
    assert done_evt["provider"] == "mock"


def test_compute_sha256_deterministic() -> None:
    """Verify compute_sha256 returns standard lowercase hex SHA-256."""
    import hashlib

    raw = b"Sample document payload for hashing."
    expected = hashlib.sha256(raw).hexdigest()
    assert compute_sha256(raw) == expected


def test_duplicate_document_upload_sha256_skips_reindexing(
    test_client: WebTestClient,
) -> None:
    """Verify duplicate upload returns existing document without new chunks."""
    content = b"Unique content for SHA-256 deduplication test across uploads."

    resp1 = test_client.post(
        "/api/documents/upload",
        files={"file": ("original.txt", io.BytesIO(content), "text/plain")},
    )
    assert resp1.status_code == 200
    doc1 = resp1.json()

    # Upload exact same content with different file name
    resp2 = test_client.post(
        "/api/documents/upload",
        files={"file": ("renamed_duplicate.txt", io.BytesIO(content), "text/plain")},
    )
    assert resp2.status_code == 200
    doc2 = resp2.json()

    # Must return identical doc_id
    assert doc1["doc_id"] == doc2["doc_id"]

    # Verify status reports only 1 document and 1 chunk (not 2)
    status_resp = test_client.get("/api/status")
    status_data = status_resp.json()
    assert status_data["total_documents"] == 1
    assert status_data["total_chunks"] == 1

    # Verify documents list contains only 1 entry
    list_resp = test_client.get("/api/documents")
    assert len(list_resp.json()) == 1


def test_knowledge_store_disk_persistence_and_reload(tmp_path: Path) -> None:
    """Verify KnowledgeStore atomically persists to disk and reloads on startup."""
    storage_file = tmp_path / "nested" / "index_state.json"
    content = b"Persistent content across restarts in KnowledgePilot AI."

    # Store 1: Ingest document and persist to disk
    store1 = KnowledgeStore(
        embedding_model=MockEmbeddingModel(dimension=4),
        generator=MockGenerator(default_response="Reloaded response."),
        storage_path=storage_file,
    )
    summary1 = store1.add_document("persisted.txt", content)

    assert storage_file.exists()
    saved_data = json.loads(storage_file.read_text(encoding="utf-8"))
    assert saved_data["version"] == 1
    assert len(saved_data["documents"]) == 1
    assert saved_data["documents"][0]["file_name"] == "persisted.txt"
    assert len(saved_data["index_entries"]) >= 1

    # Store 2: Fresh instance pointing to the same storage file
    store2 = KnowledgeStore(
        embedding_model=MockEmbeddingModel(dimension=4),
        generator=MockGenerator(default_response="Reloaded response."),
        storage_path=storage_file,
    )

    assert store2.total_documents == 1
    assert store2.total_chunks == store1.total_chunks
    assert summary1.doc_id in store2.documents

    # Query against reloaded store without re-uploading
    ans = store2.answer_question("What is persisted?")
    assert ans.has_sufficient_evidence is True
    assert len(ans.citations) >= 1
    assert ans.citations[0].file_name == "persisted.txt"

    # Ingesting the same content into store2 should be recognized as duplicate
    dup_summary = store2.add_document("copy.txt", content)
    assert dup_summary.doc_id == summary1.doc_id
    assert store2.total_documents == 1


def test_knowledge_store_clear_removes_persisted_file(tmp_path: Path) -> None:
    """Verify clear() deletes the persisted index state file."""
    storage_file = tmp_path / "index_state.json"
    store = KnowledgeStore(
        embedding_model=MockEmbeddingModel(dimension=4),
        storage_path=storage_file,
    )
    store.add_document("to_clear.txt", b"Temporary content to be deleted.")
    assert storage_file.exists()

    store.clear()
    assert not storage_file.exists()
    assert store.total_documents == 0
    assert store.total_chunks == 0
