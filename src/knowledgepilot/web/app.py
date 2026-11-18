"""FastAPI application factory and REST endpoints for KnowledgePilot AI Web UI."""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from knowledgepilot.config import Settings
from knowledgepilot.embeddings.base import BaseEmbeddingModel
from knowledgepilot.embeddings.mock import MockEmbeddingModel
from knowledgepilot.embeddings.sentence_transformer import (
    SentenceTransformerEmbedding,
)
from knowledgepilot.generation.base import BaseGenerator
from knowledgepilot.generation.errors import GenerationError
from knowledgepilot.generation.factory import create_generator
from knowledgepilot.generation.ollama import OllamaGenerator
from knowledgepilot.ingestion.errors import IngestionError
from knowledgepilot.web.state import DocumentSummary, KnowledgeStore

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"


class ChatRequest(BaseModel):
    """Payload for question answering chat endpoint."""

    query: str = Field(..., min_length=1, description="Question to answer.")
    top_k: int = Field(
        default=3, ge=1, le=10, description="Evidence chunks to retrieve."
    )
    generator: str | None = Field(
        default=None, description="Optional generator override."
    )
    model: str | None = Field(default=None, description="Optional model name override.")


class GeneratorUpdateRequest(BaseModel):
    """Payload for updating active generator provider and model."""

    provider: str = Field(
        ..., min_length=1, description="Generator provider: 'mock' or 'ollama'."
    )
    model: str | None = Field(default=None, description="Optional model name override.")
    base_url: str | None = Field(
        default=None, description="Optional Ollama base URL override."
    )


class CitationResponse(BaseModel):
    """Citation metadata for an evidence passage."""

    file_name: str
    page_number: int | None
    chunk_index: int
    source: str
    document_index: int = 1
    snippet: str = ""


class EvidenceResponse(BaseModel):
    """Retrieved evidence passage with score and metadata."""

    text: str
    score: float
    file_name: str
    page_number: int | None
    chunk_index: int
    document_index: int = 1


class ChatResponse(BaseModel):
    """Structured response from the RAG chat service."""

    query: str
    answer: str
    has_sufficient_evidence: bool
    citations: list[CitationResponse]
    evidence: list[EvidenceResponse]
    provider: str
    model: str


class StatusResponse(BaseModel):
    """System health and index status response."""

    status: str
    provider: str
    model: str
    embedding_model: str
    total_documents: int
    total_chunks: int
    ollama_available: bool


def create_web_app(
    settings: Settings | None = None,
    store: KnowledgeStore | None = None,
    embedding_model: BaseEmbeddingModel | None = None,
    generator: BaseGenerator | None = None,
) -> FastAPI:
    """Create and configure the KnowledgePilot FastAPI web application.

    Args:
        settings: Application settings (defaults to Settings.from_env()).
        store: Pre-configured KnowledgeStore instance (useful for testing).
        embedding_model: Embedding provider override.
        generator: Generator provider override.

    Returns:
        Configured FastAPI instance with REST routes and static UI assets.
    """
    resolved_settings = settings or Settings.from_env()

    if store is not None:
        knowledge_store = store
    else:
        embedder: BaseEmbeddingModel = (
            embedding_model
            if embedding_model is not None
            else (
                MockEmbeddingModel()
                if resolved_settings.is_test
                else SentenceTransformerEmbedding()
            )
        )
        gen: BaseGenerator = (
            generator
            if generator is not None
            else create_generator(
                provider=resolved_settings.generator_provider,
                base_url=resolved_settings.ollama_base_url,
                model=resolved_settings.ollama_model,
            )
        )
        storage_path = (
            None if resolved_settings.is_test else resolved_settings.storage_path
        )
        storage_path = (
            None if resolved_settings.is_test else resolved_settings.storage_path
        )
        knowledge_store = KnowledgeStore(
            embedding_model=embedder,
            generator=gen,
            storage_path=storage_path,
        )

    app = FastAPI(
        title="KnowledgePilot AI",
        description="Production-minded Grounded RAG Web Interface",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store reference on app state for tests and handlers
    app.state.store = knowledge_store
    app.state.settings = resolved_settings

    @app.get("/api/status", response_model=StatusResponse)
    def get_status() -> StatusResponse:
        current_gen = knowledge_store.generator
        ollama_ok = False
        provider_name = "mock"
        model_name = "mock"

        if isinstance(current_gen, OllamaGenerator):
            provider_name = "ollama"
            model_name = current_gen.model
            ollama_ok = current_gen.is_available()
        else:
            model_name = "deterministic-mock"
            if not resolved_settings.is_test:
                ollama_ok = OllamaGenerator(
                    base_url=resolved_settings.ollama_base_url,
                    model=resolved_settings.ollama_model,
                ).is_available()

        embed_name = type(knowledge_store.embedding_model).__name__

        return StatusResponse(
            status="healthy",
            provider=provider_name,
            model=model_name,
            embedding_model=embed_name,
            total_documents=knowledge_store.total_documents,
            total_chunks=knowledge_store.total_chunks,
            ollama_available=ollama_ok,
        )

    @app.post("/api/generator", response_model=StatusResponse)
    def set_generator_endpoint(req: GeneratorUpdateRequest) -> StatusResponse:
        clean_provider = req.provider.lower().strip()
        if clean_provider not in {"mock", "ollama"}:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported generator provider '{req.provider}'. "
                    "Supported providers: 'mock', 'ollama'."
                ),
            )

        model_name = (
            req.model.strip()
            if req.model and req.model.strip()
            else (
                "llama3.2:3b"
                if clean_provider == "ollama"
                else resolved_settings.ollama_model
            )
        )
        base_url = (
            req.base_url.rstrip("/")
            if req.base_url and req.base_url.strip()
            else resolved_settings.ollama_base_url
        )

        try:
            knowledge_store.set_generator(
                provider=clean_provider,
                base_url=base_url,
                model=model_name,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return get_status()

    @app.get("/api/documents", response_model=list[DocumentSummary])
    def list_documents() -> list[DocumentSummary]:
        return knowledge_store.list_documents()

    @app.post("/api/documents/upload", response_model=DocumentSummary)
    async def upload_document(
        file: UploadFile = File(...),  # noqa: B008
    ) -> DocumentSummary:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename missing from upload.")

        data = await file.read()
        try:
            summary = knowledge_store.add_document(file.filename, data)
            return summary
        except (ValueError, IngestionError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Upload processing failed for %s", file.filename)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process document: {exc}",
            ) from exc

    @app.delete("/api/documents")
    def clear_documents() -> dict[str, str]:
        knowledge_store.clear()
        return {"status": "cleared", "message": "Knowledge store reset successfully."}

    @app.post("/api/chat", response_model=ChatResponse)
    def chat(req: ChatRequest) -> ChatResponse:
        if req.generator:
            clean_provider = req.generator.lower().strip()
            if clean_provider not in {"mock", "ollama"}:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Unsupported generator provider '{req.generator}'. "
                        "Supported providers: 'mock', 'ollama'."
                    ),
                )
            selected_model = (
                req.model.strip()
                if req.model and req.model.strip()
                else (
                    "llama3.2:3b"
                    if clean_provider == "ollama"
                    else resolved_settings.ollama_model
                )
            )
            try:
                knowledge_store.set_generator(
                    provider=clean_provider,
                    base_url=resolved_settings.ollama_base_url,
                    model=selected_model,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            rag_ans = knowledge_store.answer_question(req.query, top_k=req.top_k)
        except GenerationError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Chat query failed for: %s", req.query)
            raise HTTPException(
                status_code=500,
                detail=f"Generation failed: {exc}",
            ) from exc

        provider_name = rag_ans.provider
        model_name = rag_ans.model

        citations = [
            CitationResponse(
                file_name=c.file_name,
                page_number=c.page_number,
                chunk_index=c.chunk_index,
                source=c.source,
                document_index=c.document_index,
                snippet=c.snippet,
            )
            for c in rag_ans.citations
        ]

        evidence = [
            EvidenceResponse(
                text=ev.text,
                score=ev.score,
                file_name=ev.file_name,
                page_number=ev.page_number,
                chunk_index=ev.chunk_index,
                document_index=idx,
            )
            for idx, ev in enumerate(rag_ans.evidence, start=1)
        ]

        return ChatResponse(
            query=rag_ans.query,
            answer=rag_ans.answer,
            has_sufficient_evidence=rag_ans.has_sufficient_evidence,
            citations=citations,
            evidence=evidence,
            provider=provider_name,
            model=model_name,
        )

    @app.post("/api/chat/stream")
    def chat_stream(req: ChatRequest) -> StreamingResponse:
        """Stream grounded RAG tokens and final citations via Server-Sent Events (SSE).

        Each SSE event is a JSON object on a ``data:`` line.  Three event types
        are emitted:

        * ``{"type": "token",  "token": "<text>"}`` — incremental LLM token.
        * ``{"type": "done",   "answer": "...", "citations": [...],
          "evidence": [...], "provider": "...", "model": "...",
          "has_sufficient_evidence": true}`` — final consolidated payload.
        * ``{"type": "error",  "detail": "<message>"}`` — generation error.
        """

        def _resolve_generator() -> None:
            """Optionally switch generator/model for this request."""
            if not req.generator:
                return
            clean = req.generator.lower().strip()
            if clean not in {"mock", "ollama"}:
                return
            selected_model = (
                req.model.strip()
                if req.model and req.model.strip()
                else (
                    "llama3.2:3b"
                    if clean == "ollama"
                    else resolved_settings.ollama_model
                )
            )
            with contextlib.suppress(ValueError):
                knowledge_store.set_generator(
                    provider=clean,
                    base_url=resolved_settings.ollama_base_url,
                    model=selected_model,
                )

        def event_stream() -> Iterator[str]:
            try:
                _resolve_generator()

                # 1. Retrieve evidence (uses existing thread-safe KnowledgeStore)
                from knowledgepilot.generation.errors import GenerationError as GenError
                from knowledgepilot.generation.prompts import format_grounded_qa_prompt
                from knowledgepilot.rag.service import INSUFFICIENT_EVIDENCE_ANSWER

                with knowledge_store._lock:
                    assert knowledge_store._retriever is not None
                    evidence_results = knowledge_store._retriever.search(
                        req.query.strip(), top_k=req.top_k
                    )
                    gen = knowledge_store.generator

                provider_name = getattr(gen, "provider", type(gen).__name__.lower())
                model_name = str(getattr(gen, "model", "mock"))

                from knowledgepilot.generation.mock import MockGenerator as _Mock
                from knowledgepilot.generation.ollama import OllamaGenerator as _Ollama

                if isinstance(gen, _Ollama):
                    provider_name = "ollama"
                    model_name = gen.model
                elif isinstance(gen, _Mock):
                    provider_name = "mock"
                    model_name = "mock"

                if not evidence_results:
                    done_payload: dict[str, Any] = {
                        "type": "done",
                        "query": req.query,
                        "answer": INSUFFICIENT_EVIDENCE_ANSWER,
                        "has_sufficient_evidence": False,
                        "citations": [],
                        "evidence": [],
                        "provider": provider_name,
                        "model": model_name,
                    }
                    yield f"data: {json.dumps(done_payload)}\n\n"
                    return

                # Build citations list
                citations_data: list[dict[str, Any]] = []
                for idx, item in enumerate(evidence_results, start=1):
                    snippet = item.text.strip().replace("\n", " ")
                    if len(snippet) > 180:
                        snippet = snippet[:177] + "..."
                    citations_data.append(
                        {
                            "file_name": item.file_name,
                            "page_number": item.page_number,
                            "chunk_index": item.chunk_index,
                            "source": item.source,
                            "document_index": idx,
                            "snippet": snippet,
                        }
                    )

                evidence_data: list[dict[str, Any]] = [
                    {
                        "text": item.text,
                        "score": item.score,
                        "file_name": item.file_name,
                        "page_number": item.page_number,
                        "chunk_index": item.chunk_index,
                        "document_index": idx,
                    }
                    for idx, item in enumerate(evidence_results, start=1)
                ]

                # 2. Format grounded prompt
                prompt = format_grounded_qa_prompt(
                    query=req.query.strip(),
                    evidence=evidence_results,
                )

                # 3. Stream tokens via generate_stream()
                streamed_tokens: list[str] = []
                try:
                    for token in gen.generate_stream(prompt):
                        streamed_tokens.append(token)
                        tok_payload = json.dumps({"type": "token", "token": token})
                        yield f"data: {tok_payload}\n\n"
                except GenError as exc:
                    err_payload = json.dumps({"type": "error", "detail": str(exc)})
                    yield f"data: {err_payload}\n\n"
                    return

                full_answer = "".join(streamed_tokens).strip()

                # 4. Final consolidated SSE event with citations + evidence
                final_payload: dict[str, Any] = {
                    "type": "done",
                    "query": req.query,
                    "answer": full_answer,
                    "has_sufficient_evidence": bool(full_answer),
                    "citations": citations_data,
                    "evidence": evidence_data,
                    "provider": provider_name,
                    "model": model_name,
                }
                yield f"data: {json.dumps(final_payload)}\n\n"

            except Exception as exc:
                logger.exception("Streaming chat error for query: %s", req.query)
                yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)})}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Static assets and index.html serving
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def serve_index() -> Any:
        index_file = STATIC_DIR / "index.html"
        if not index_file.exists():
            return {"message": "KnowledgePilot AI Web API running. Frontend loading..."}
        return FileResponse(index_file)

    return app
