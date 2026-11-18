"""RAG service orchestrating document retrieval and grounded question answering."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from knowledgepilot.embeddings.base import BaseEmbeddingModel
from knowledgepilot.embeddings.sentence_transformer import (
    SentenceTransformerEmbedding,
)
from knowledgepilot.generation.base import BaseGenerator
from knowledgepilot.generation.errors import GenerationError
from knowledgepilot.generation.factory import create_generator
from knowledgepilot.generation.mock import MockGenerator
from knowledgepilot.generation.ollama import OllamaGenerator
from knowledgepilot.generation.prompts import format_grounded_qa_prompt
from knowledgepilot.ingestion.chunker import TextChunker
from knowledgepilot.ingestion.loaders import load_document
from knowledgepilot.rag.models import Citation, RagAnswer
from knowledgepilot.retrieval.index import VectorIndex
from knowledgepilot.retrieval.retriever import SemanticRetriever

INSUFFICIENT_EVIDENCE_ANSWER = (
    "I do not have sufficient information in the provided documents "
    "to answer this question."
)

_INSUFFICIENT_PHRASES = (
    "do not have sufficient information",
    "insufficient information",
    "not enough information",
    "insufficient evidence",
    "cannot answer based on the provided",
    "not mentioned in the provided",
)


class RagService:
    """Service orchestrating semantic retrieval and grounded answer generation."""

    def __init__(
        self,
        retriever: SemanticRetriever,
        generator: BaseGenerator,
        system_instruction: str | None = None,
    ) -> None:
        """Initialize the RAG service.

        Args:
            retriever: Configured SemanticRetriever for finding evidence chunks.
            generator: BaseGenerator implementation for producing answers.
            system_instruction: Optional system instruction override.
        """
        self.retriever = retriever
        self.generator = generator
        self.system_instruction = system_instruction

    def answer_question(
        self,
        query: str,
        top_k: int = 3,
        **kwargs: Any,
    ) -> RagAnswer:
        """Retrieve relevant context and generate a grounded answer.

        Args:
            query: The user's input question.
            top_k: Maximum number of evidence chunks to retrieve.
            **kwargs: Extra arguments forwarded to generator.generate().

        Returns:
            A structured RagAnswer containing the text answer, citations with
            filenames and page numbers, and the raw evidence chunks.

        Raises:
            ValueError: If query is empty or top_k <= 0.
            GenerationError: If the answer generator fails.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty or whitespace-only.")

        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        evidence = self.retriever.search(query.strip(), top_k=top_k)

        gen = self.generator
        if isinstance(gen, OllamaGenerator):
            provider_name = "ollama"
            model_name = gen.model
        elif isinstance(gen, MockGenerator):
            provider_name = "mock"
            model_name = getattr(gen, "model", "mock")
        else:
            provider_name = getattr(
                gen,
                "provider",
                type(gen).__name__.lower().replace("generator", ""),
            )
            model_name = getattr(gen, "model", "default")

        if not evidence:
            return RagAnswer(
                query=query.strip(),
                answer=INSUFFICIENT_EVIDENCE_ANSWER,
                citations=[],
                evidence=[],
                has_sufficient_evidence=False,
                provider=provider_name,
                model=model_name,
            )

        citations: list[Citation] = []
        for idx, item in enumerate(evidence, start=1):
            snippet = item.text.strip().replace("\n", " ")
            if len(snippet) > 180:
                snippet = snippet[:177] + "..."
            citations.append(
                Citation(
                    file_name=item.file_name,
                    page_number=item.page_number,
                    source=item.source,
                    chunk_index=item.chunk_index,
                    document_index=idx,
                    snippet=snippet,
                )
            )

        prompt = format_grounded_qa_prompt(
            query=query.strip(),
            evidence=evidence,
            system_instruction=self.system_instruction,
        )

        try:
            raw_answer = self.generator.generate(prompt, **kwargs)
        except GenerationError:
            raise
        except Exception as exc:
            raise GenerationError(f"Unexpected generation failure: {exc}") from exc

        lower_answer = raw_answer.strip().lower()
        starts_with_insufficient = any(
            lower_answer.startswith(p)
            for p in (
                "i do not have sufficient information",
                "insufficient information",
                "not enough information",
                "insufficient evidence",
                "cannot answer based on the provided",
                "there is insufficient information",
            )
        )
        has_citations = bool(
            re.search(r"\[(?:Document|Doc)\s*\d+", raw_answer, re.IGNORECASE)
        )

        if starts_with_insufficient and not has_citations:
            has_sufficient = False
        elif starts_with_insufficient and has_citations:
            has_sufficient = len(raw_answer.split()) > 35
        elif has_citations:
            has_sufficient = True
        else:
            has_sufficient = not any(p in lower_answer for p in _INSUFFICIENT_PHRASES)

        return RagAnswer(
            query=query.strip(),
            answer=raw_answer,
            citations=citations,
            evidence=evidence,
            has_sufficient_evidence=has_sufficient,
            provider=provider_name,
            model=model_name,
        )


def ask_document(
    doc_path: str | Path,
    query: str,
    *,
    top_k: int = 3,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    embedding_model: BaseEmbeddingModel | None = None,
    generator: BaseGenerator | None = None,
    generator_provider: str = "mock",
    ollama_base_url: str | None = None,
    ollama_model: str | None = None,
    system_instruction: str | None = None,
) -> RagAnswer:
    """Ingest a local document, chunk it, index embeddings, and answer a question.

    Args:
        doc_path: Path to a local .txt or .pdf document.
        query: Question to ask about the document.
        top_k: Maximum number of evidence passages to retrieve.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Number of overlapping characters between adjacent chunks.
        embedding_model: Optional embedding model instance. Defaults to
            SentenceTransformerEmbedding('all-MiniLM-L6-v2').
        generator: Optional pre-configured generator instance. If None,
            generator_provider is used to instantiate one.
        generator_provider: Backend provider name ('mock' or 'ollama').
            Defaults to 'mock'.
        ollama_base_url: Optional Ollama base URL override (used when provider
            is 'ollama').
        ollama_model: Optional Ollama model name override (used when provider
            is 'ollama').
        system_instruction: Optional system instruction override.

    Returns:
        Structured RagAnswer with cited filenames, page numbers, and evidence.
    """
    documents = load_document(doc_path)
    embedder = (
        embedding_model
        if embedding_model is not None
        else SentenceTransformerEmbedding()
    )
    if generator is not None:
        gen = generator
    else:
        gen = create_generator(
            provider=generator_provider,
            base_url=ollama_base_url,
            model=ollama_model,
        )

    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    retriever = SemanticRetriever(embedding_model=embedder, index=VectorIndex())
    retriever.index_documents(documents, chunker=chunker)
    service = RagService(
        retriever=retriever,
        generator=gen,
        system_instruction=system_instruction,
    )
    return service.answer_question(query, top_k=top_k)
