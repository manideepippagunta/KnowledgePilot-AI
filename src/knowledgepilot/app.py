"""Application entry point and CLI for KnowledgePilot AI."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from knowledgepilot.config import Settings
from knowledgepilot.embeddings.base import BaseEmbeddingModel
from knowledgepilot.embeddings.mock import MockEmbeddingModel
from knowledgepilot.embeddings.sentence_transformer import (
    SentenceTransformerEmbedding,
)
from knowledgepilot.generation.factory import create_generator
from knowledgepilot.rag.service import ask_document

logger = logging.getLogger(__name__)


def configure_logging(level: str) -> None:
    """Configure root logging at the given level name.

    Safe to call multiple times — re-configures the root logger.
    """
    numeric = getattr(logging, level.upper(), None)
    if not isinstance(numeric, int):
        raise ValueError(f"Invalid log level: {level!r}")
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        force=True,
    )


_API_KEY_REQUIRED_PROVIDERS: frozenset[str] = frozenset({"openai"})


def create_app(settings: Settings | None = None) -> Settings:
    """Create / validate application state.

    Args:
        settings: Optional pre-built settings (useful for testing).
                  If None, settings are loaded from the environment.

    Returns:
        The validated Settings instance.
    """
    resolved = settings if settings is not None else Settings.from_env()
    configure_logging(resolved.log_level)
    logger.info(
        "KnowledgePilot starting | env=%s host=%s port=%d log_level=%s",
        resolved.env,
        resolved.host,
        resolved.port,
        resolved.log_level,
    )
    if (
        resolved.generator_provider in _API_KEY_REQUIRED_PROVIDERS
        and resolved.openai_api_key is None
    ):
        logger.warning(
            "No LLM API key found in environment (OPENAI_API_KEY). "
            "Provider '%s' generation features will be unavailable "
            "until a key is configured.",
            resolved.generator_provider,
        )
    return resolved


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Can run in default application service mode or execute a grounded
    question-answering query against a local document.

    Args:
        argv: Optional argument list (defaults to sys.argv[1:]).

    Returns:
        Process exit code (0 on success, non-zero on failure).
    """
    args_list = argv if argv is not None else sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="knowledgepilot",
        description="KnowledgePilot AI - Production-minded RAG Application",
    )
    parser.add_argument(
        "--doc",
        "-d",
        type=str,
        help="Path to a local document (.txt or .pdf) to query against.",
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        help="The question to ask against the document.",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=3,
        help="Number of relevant evidence passages to retrieve (default: 3).",
    )
    parser.add_argument(
        "--mock-embeddings",
        action="store_true",
        help="Use deterministic mock embeddings instead of sentence-transformers.",
    )
    parser.add_argument(
        "--generator",
        "-g",
        type=str,
        choices=["mock", "ollama"],
        default=None,
        help="Generation provider to use ('mock' or 'ollama'). Defaults to 'mock'.",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default=None,
        help="Endpoint for local Ollama service (default: http://localhost:11434).",
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default=None,
        help="Ollama model name (default: llama3.2:3b).",
    )
    parser.add_argument(
        "--serve",
        "-s",
        action="store_true",
        help="Launch the interactive web UI server.",
    )
    parser.add_argument(
        "--storage-path",
        type=str,
        default=None,
        help=(
            "Path for atomic disk index state persistence "
            "(default: .knowledgepilot/index_state.json in server mode)."
        ),
    )

    try:
        parsed, _ = parser.parse_known_args(args_list)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2

    # Web server mode: launch the FastAPI web application
    if parsed.serve:
        try:
            settings = create_app()
        except ValueError as exc:
            print(f"Configuration error: {exc}", file=sys.stderr)
            return 2

        import uvicorn

        from knowledgepilot.web.app import create_web_app
        from knowledgepilot.web.state import KnowledgeStore

        gen_provider = parsed.generator or settings.generator_provider
        ollama_url = parsed.ollama_url or settings.ollama_base_url
        ollama_model = parsed.ollama_model or settings.ollama_model
        initial_gen = create_generator(
            provider=gen_provider,
            base_url=ollama_url,
            model=ollama_model,
        )
        storage_path = (
            parsed.storage_path
            or settings.storage_path
            or ".knowledgepilot/index_state.json"
        )
        embedder: BaseEmbeddingModel = (
            MockEmbeddingModel()
            if parsed.mock_embeddings
            else SentenceTransformerEmbedding()
        )
        knowledge_store = KnowledgeStore(
            embedding_model=embedder,
            generator=initial_gen,
            storage_path=storage_path,
        )

        web_app = create_web_app(
            settings=settings,
            store=knowledge_store,
        )
        print("=" * 68)
        print("KnowledgePilot AI - Web Application Server")
        print("=" * 68)
        print(f"Access Web UI at: http://{settings.host}:{settings.port}")
        print(f"Provider        : {gen_provider}")
        print(f"Ollama Endpoint : {ollama_url} (model: {ollama_model})")
        print(f"Storage Path    : {storage_path}")
        print("=" * 68)
        uvicorn.run(
            web_app,
            host=settings.host,
            port=settings.port,
            log_level=settings.log_level.lower(),
        )
        return 0

    # Query mode: ask question against a specific document
    if parsed.doc is not None:
        if not parsed.query:
            print(
                "Error: --query (-q) is required when --doc is specified.",
                file=sys.stderr,
            )
            return 2

        doc_path = Path(parsed.doc)
        if not doc_path.exists() or not doc_path.is_file():
            print(
                f"Error: Document file not found: {parsed.doc}",
                file=sys.stderr,
            )
            return 2

        try:
            settings = create_app()
        except ValueError as exc:
            print(f"Configuration error: {exc}", file=sys.stderr)
            return 2

        embedder = (
            MockEmbeddingModel()
            if parsed.mock_embeddings
            else SentenceTransformerEmbedding()
        )

        chosen_provider = (
            parsed.generator
            if parsed.generator is not None
            else settings.generator_provider
        )
        resolved_url = (
            parsed.ollama_url
            if parsed.ollama_url is not None
            else settings.ollama_base_url
        )
        resolved_model = (
            parsed.ollama_model
            if parsed.ollama_model is not None
            else settings.ollama_model
        )

        try:
            generator = create_generator(
                provider=chosen_provider,
                base_url=resolved_url,
                model=resolved_model,
            )
        except ValueError as exc:
            print(f"Configuration error: {exc}", file=sys.stderr)
            return 2

        try:
            result = ask_document(
                doc_path=doc_path,
                query=parsed.query,
                top_k=parsed.top_k,
                embedding_model=embedder,
                generator=generator,
            )
        except Exception as exc:
            print(f"Error executing RAG query: {exc}", file=sys.stderr)
            return 1

        print("=" * 68)
        print("KnowledgePilot AI - Grounded RAG Answer")
        print("=" * 68)
        print(f"Document : {doc_path.name}")
        print(f"Query    : {result.query}")
        print("-" * 68)
        print("Answer:")
        print(result.answer)
        print()
        if chosen_provider == "mock":
            print(
                "[Note: Answer synthesized by evidence-grounded MockGenerator, "
                "not an autonomous LLM.]"
            )
        else:
            print(
                f"[Provider: Local Ollama LLM | Model: {resolved_model} | "
                f"Endpoint: {resolved_url}]"
            )
        print("-" * 68)
        print(f"Citations ({len(result.citations)}):")
        for cit in result.citations:
            p_text = f"Page {cit.page_number}" if cit.page_number else "Page N/A"
            print(f"  * [Doc {cit.document_index}] {cit.file_name} ({p_text})")
        print()
        print(f"Evidence Chunks ({len(result.evidence)}):")
        for i, ev in enumerate(result.evidence, start=1):
            p_str = f"Page {ev.page_number}" if ev.page_number else "Page N/A"
            print(f"  [Document {i}] Score: {ev.score:.4f} | {ev.file_name} ({p_str})")
            print(f"      {ev.text[:120]}...")
        print("=" * 68)
        return 0

    # Default application banner mode
    try:
        settings = create_app()
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover — defensive
        print(f"Unexpected startup error: {exc}", file=sys.stderr)
        return 1

    print(
        f"KnowledgePilot AI is running | env={settings.env} "
        f"host={settings.host} port={settings.port}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
