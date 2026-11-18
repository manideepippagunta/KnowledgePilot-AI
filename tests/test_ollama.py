"""Unit tests for local Ollama generator provider and CLI integration (Phase 6).

All tests in this module run offline and deterministically by mocking
standard library HTTP responses. No network calls or model downloads occur.
"""

from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from knowledgepilot.app import main
from knowledgepilot.embeddings.mock import MockEmbeddingModel
from knowledgepilot.generation.errors import GenerationError
from knowledgepilot.generation.factory import create_generator
from knowledgepilot.generation.mock import MockGenerator
from knowledgepilot.generation.ollama import OllamaGenerator
from knowledgepilot.ingestion.models import Chunk, ChunkMetadata
from knowledgepilot.rag import ask_document
from knowledgepilot.rag.service import INSUFFICIENT_EVIDENCE_ANSWER, RagService
from knowledgepilot.retrieval.models import SearchResult


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


def _mock_http_response(
    data: dict[str, Any] | bytes | str,
    status: int = 200,
) -> MagicMock:
    """Helper creating a mock urllib.request.urlopen response context manager."""
    mock_resp = MagicMock()
    mock_resp.status = status
    if isinstance(data, dict):
        raw = json.dumps(data).encode("utf-8")
    elif isinstance(data, str):
        raw = data.encode("utf-8")
    else:
        raw = data
    mock_resp.read.return_value = raw
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None
    return mock_resp


# ---------------------------------------------------------------------------
# OllamaGenerator Unit Tests
# ---------------------------------------------------------------------------


def test_ollama_generator_defaults() -> None:
    gen = OllamaGenerator()
    assert gen.base_url == "http://localhost:11434"
    assert gen.model == "llama3.2:3b"
    assert gen.timeout == 30.0


def test_ollama_generator_custom_config() -> None:
    gen = OllamaGenerator(
        base_url="http://192.168.1.100:11434/",
        model="mistral:7b",
        timeout=15.0,
    )
    assert gen.base_url == "http://192.168.1.100:11434"
    assert gen.model == "mistral:7b"
    assert gen.timeout == 15.0


def test_ollama_generator_empty_prompt_raises() -> None:
    gen = OllamaGenerator()
    with pytest.raises(ValueError, match="cannot be empty"):
        gen.generate("")

    with pytest.raises(ValueError, match="cannot be empty"):
        gen.generate("   \n\t  ")


def test_ollama_generator_successful_completion() -> None:
    gen = OllamaGenerator(base_url="http://localhost:11434", model="llama3.2:1b")
    expected_answer = "KnowledgePilot AI indexes documents into vector space."
    mock_resp = _mock_http_response({"response": expected_answer, "done": True})

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        ans = gen.generate("Explain KnowledgePilot AI.")
        assert ans == expected_answer
        assert mock_urlopen.call_count == 1
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "http://localhost:11434/api/generate"
        assert req.headers["Content-type"] == "application/json"
        payload = json.loads(req.data.decode("utf-8"))
        assert payload["model"] == "llama3.2:1b"
        assert payload["stream"] is False
        assert payload["options"]["temperature"] == 0.0


def test_ollama_generator_http_error_404_model_not_found() -> None:
    gen = OllamaGenerator(model="custom-model")
    http_err = urllib.error.HTTPError(
        url="http://localhost:11434/api/generate",
        code=404,
        msg="Not Found",
        hdrs=None,  # type: ignore[arg-type]
        fp=io.BytesIO(b"model not found"),
    )

    with patch("urllib.request.urlopen", side_effect=http_err):
        with pytest.raises(GenerationError, match="model 'custom-model' was not found"):
            gen.generate("Some prompt")


def test_ollama_generator_http_error_500() -> None:
    gen = OllamaGenerator()
    http_err = urllib.error.HTTPError(
        url="http://localhost:11434/api/generate",
        code=500,
        msg="Internal Server Error",
        hdrs=None,  # type: ignore[arg-type]
        fp=io.BytesIO(b"server crash"),
    )

    with patch("urllib.request.urlopen", side_effect=http_err):
        with pytest.raises(GenerationError, match="Ollama HTTP error \\(500\\)"):
            gen.generate("Some prompt")


def test_ollama_generator_url_error_connection_refused() -> None:
    gen = OllamaGenerator()
    url_err = urllib.error.URLError("Connection refused")

    with patch("urllib.request.urlopen", side_effect=url_err):
        with pytest.raises(GenerationError, match="Could not connect to Ollama"):
            gen.generate("Some prompt")


def test_ollama_generator_timeout_error() -> None:
    gen = OllamaGenerator(timeout=5.0)

    with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
        with pytest.raises(GenerationError, match=r"timed out after 5\.0s"):
            gen.generate("Some prompt")


def test_ollama_generator_os_error() -> None:
    gen = OllamaGenerator()

    with patch("urllib.request.urlopen", side_effect=OSError("Network is unreachable")):
        with pytest.raises(
            GenerationError, match="I/O error communicating with Ollama"
        ):
            gen.generate("Some prompt")


def test_ollama_generator_malformed_json_response() -> None:
    gen = OllamaGenerator()
    mock_resp = _mock_http_response("<html>Bad Gateway</html>")

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with pytest.raises(GenerationError, match="Malformed JSON received"):
            gen.generate("Some prompt")


def test_ollama_generator_non_dict_json_response() -> None:
    gen = OllamaGenerator()
    mock_resp = _mock_http_response('["array", "not", "object"]')

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with pytest.raises(GenerationError, match="expected a JSON object"):
            gen.generate("Some prompt")


def test_ollama_generator_missing_response_field() -> None:
    gen = OllamaGenerator()
    mock_resp = _mock_http_response({"done": True, "eval_count": 42})

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with pytest.raises(GenerationError, match="missing 'response' text field"):
            gen.generate("Some prompt")


def test_ollama_is_available_true() -> None:
    gen = OllamaGenerator()
    mock_resp = _mock_http_response({"models": []}, status=200)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        assert gen.is_available() is True


def test_ollama_is_available_false_on_connection_error() -> None:
    gen = OllamaGenerator()
    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.URLError("refused"),
    ):
        assert gen.is_available() is False


def test_mock_generator_generate_stream() -> None:
    gen = MockGenerator(default_response="KnowledgePilot AI streams responses.")
    tokens = list(gen.generate_stream("Any prompt"))
    assert "".join(tokens) == "KnowledgePilot AI streams responses."


def test_ollama_generator_generate_stream_success() -> None:
    gen = OllamaGenerator()
    lines = [
        b'{"response": "Ground", "done": false}\n',
        b'{"response": "ed RAG", "done": true}\n',
    ]
    mock_resp = MagicMock()
    mock_resp.__iter__.return_value = lines
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None

    with patch("urllib.request.urlopen", return_value=mock_resp):
        tokens = list(gen.generate_stream("What is this?"))

    assert tokens == ["Ground", "ed RAG"]


def test_ollama_generator_generate_stream_empty_prompt() -> None:
    gen = OllamaGenerator()
    with pytest.raises(ValueError, match="Prompt must not be empty"):
        list(gen.generate_stream("   "))


def test_ollama_generator_generate_stream_connection_error() -> None:
    gen = OllamaGenerator()
    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.URLError("Connection refused"),
    ):
        with pytest.raises(GenerationError, match="Could not connect to Ollama"):
            list(gen.generate_stream("Test prompt"))


# ---------------------------------------------------------------------------
# Factory Tests
# ---------------------------------------------------------------------------


def test_create_generator_mock() -> None:
    gen = create_generator("mock")
    assert isinstance(gen, MockGenerator)


def test_create_generator_ollama() -> None:
    gen = create_generator(
        "ollama",
        base_url="http://custom:11434",
        model="phi3",
    )
    assert isinstance(gen, OllamaGenerator)
    assert gen.base_url == "http://custom:11434"
    assert gen.model == "phi3"


def test_create_generator_unsupported_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported generator provider 'openai'"):
        create_generator("openai")


# ---------------------------------------------------------------------------
# RAG Service with Ollama Provider Tests
# ---------------------------------------------------------------------------


def test_rag_service_with_ollama_evidence_grounding_and_citations() -> None:
    """Verify that RagService correctly invokes Ollama and preserves citations."""
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [
        _make_search_result(
            text="Vector embeddings map words to continuous high-dimensional space.",
            score=0.92,
            file_name="manual.pdf",
            page_number=4,
        )
    ]

    ollama_gen = OllamaGenerator()
    expected_ans = (
        "According to [Document 1], vector embeddings map words to continuous space."
    )
    mock_resp = _mock_http_response({"response": expected_ans, "done": True})

    service = RagService(retriever=mock_retriever, generator=ollama_gen)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        answer = service.answer_question("What do vector embeddings do?")

    assert answer.has_sufficient_evidence is True
    assert answer.answer == expected_ans
    assert len(answer.citations) == 1
    assert answer.citations[0].file_name == "manual.pdf"
    assert answer.citations[0].page_number == 4
    assert len(answer.evidence) == 1


def test_rag_service_with_ollama_insufficient_evidence_response() -> None:
    """Verify uncertainty response sets has_sufficient_evidence to False."""
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [
        _make_search_result(
            text="Unrelated text about cooking recipes.",
            score=0.15,
            file_name="cookbook.txt",
            page_number=1,
        )
    ]

    ollama_gen = OllamaGenerator()
    uncertainty_reply = (
        "I do not have sufficient information in the provided documents "
        "to answer this question."
    )
    mock_resp = _mock_http_response({"response": uncertainty_reply, "done": True})

    service = RagService(retriever=mock_retriever, generator=ollama_gen)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        answer = service.answer_question("What is quantum computing?")

    assert answer.has_sufficient_evidence is False
    assert (
        "do not have sufficient information" in answer.answer.lower()
        or "insufficient" in answer.answer.lower()
    )


def test_rag_service_no_evidence_returns_uncertainty_without_llm_call() -> None:
    """If retriever yields no evidence, return uncertainty without hitting Ollama."""
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = []

    ollama_gen = OllamaGenerator()
    service = RagService(retriever=mock_retriever, generator=ollama_gen)

    with patch("urllib.request.urlopen") as mock_urlopen:
        ans = service.answer_question("Any question?")
        assert ans.has_sufficient_evidence is False
        assert ans.answer == INSUFFICIENT_EVIDENCE_ANSWER
        assert ans.citations == []
        mock_urlopen.assert_not_called()


def test_ask_document_with_ollama_provider(tmp_path: Path) -> None:
    doc_file = tmp_path / "rag_guide.txt"
    doc_file.write_text(
        "Semantic search uses cosine similarity over dense vector embeddings.",
        encoding="utf-8",
    )

    mock_resp = _mock_http_response(
        {
            "response": "Cosine similarity measures vector alignment.",
            "done": True,
        }
    )

    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = ask_document(
            doc_path=doc_file,
            query="What measures vector alignment?",
            embedding_model=MockEmbeddingModel(dimension=4),
            generator_provider="ollama",
            ollama_base_url="http://localhost:11434",
            ollama_model="llama3.2:1b",
        )

    assert result.has_sufficient_evidence is True
    assert result.answer == "Cosine similarity measures vector alignment."
    assert result.cited_files == ["rag_guide.txt"]
    assert len(result.evidence) >= 1


# ---------------------------------------------------------------------------
# CLI Integration Tests for Local LLM Option
# ---------------------------------------------------------------------------


def test_cli_with_ollama_generator_success(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    doc_file = tmp_path / "notes.txt"
    doc_file.write_text(
        "KnowledgePilot supports local LLMs via Ollama.",
        encoding="utf-8",
    )

    mock_resp = _mock_http_response(
        {
            "response": "Ollama is supported for local LLM generation.",
            "done": True,
        }
    )

    with patch("urllib.request.urlopen", return_value=mock_resp):
        code = main(
            [
                "--doc",
                str(doc_file),
                "--query",
                "What is supported for local LLMs?",
                "--generator",
                "ollama",
                "--mock-embeddings",
            ]
        )

    assert code == 0
    out = capsys.readouterr().out
    assert "Provider: Local Ollama LLM" in out
    assert "llama3.2:3b" in out
    assert "Ollama is supported for local LLM generation." in out
    assert "Citations (1):" in out
    assert "notes.txt" in out


def test_cli_with_ollama_generator_connection_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    doc_file = tmp_path / "notes.txt"
    doc_file.write_text("Sample content.", encoding="utf-8")

    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.URLError("Connection refused"),
    ):
        code = main(
            [
                "--doc",
                str(doc_file),
                "--query",
                "What is here?",
                "--generator",
                "ollama",
                "--mock-embeddings",
            ]
        )

    assert code == 1
    err = capsys.readouterr().err
    assert "Error executing RAG query:" in err
    assert "Could not connect to Ollama" in err


# ---------------------------------------------------------------------------
# Optional Real Integration Test for Live Ollama
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_real_live_ollama_integration_if_available(tmp_path: Path) -> None:
    """Live Ollama test: only executed when `pytest -m integration` is invoked."""
    gen = OllamaGenerator()
    if not gen.is_available():
        pytest.skip(
            "Local Ollama service is not reachable at http://localhost:11434. "
            "Start Ollama (`ollama serve`) to run this test."
        )

    doc_file = tmp_path / "live_doc.txt"
    doc_file.write_text(
        "The sky on Mars appears butterscotch or reddish-pink during the day.",
        encoding="utf-8",
    )

    result = ask_document(
        doc_path=doc_file,
        query="What color is the sky on Mars during the day?",
        generator_provider="ollama",
    )

    assert len(result.citations) >= 1
    assert result.cited_files == ["live_doc.txt"]
    assert len(result.answer) > 0
