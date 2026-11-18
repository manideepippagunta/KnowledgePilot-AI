# KnowledgePilot AI

Production-minded Retrieval-Augmented Generation (RAG) application — **Phase 6: Real Local LLM Generation**.

This repository contains the complete, production-minded end-to-end RAG pipeline:
- **Foundation & Configuration**: Type-safe settings from environment variables (`Settings`).
- **Document Ingestion**: Parsing for `.txt` and `.pdf` files with page-level metadata preservation (`load_document`, `TxtLoader`, `PdfLoader`).
- **Text Chunking**: Sliding-window chunker with overlap and character/page offset tracking (`TextChunker`).
- **Dense Embeddings**: Provider-independent interface (`BaseEmbeddingModel`), local open-source sentence-transformers with lazy loading (`SentenceTransformerEmbedding`), and deterministic mock embeddings (`MockEmbeddingModel`).
- **Vector Search**: In-memory dense vector store with cosine similarity, vector dimension validation, and deterministic tie-breaking (`VectorIndex`).
- **Semantic Retrieval**: High-level chunk and document retriever returning structured matches with source and page metadata (`SemanticRetriever`, `SearchResult`).
- **Local LLM Generation (Ollama)**: Provider-independent generation interface (`BaseGenerator`, `OllamaGenerator`, `create_generator`) communicating with local Ollama instances via standard-library HTTP. No external paid APIs or secret keys required.
- **Deterministic Mock Generation**: Zero-download, offline mock generator (`MockGenerator`) kept as the default for lightning-fast unit tests and offline environments.
- **Strict Evidence Grounding**: Prompts and pipeline instruct models to answer strictly from provided evidence chunks, preserve document filenames and page citations, and return an explicit uncertainty acknowledgment when evidence is insufficient.
- **RAG Orchestration**: End-to-end question answering pipeline returning structured answers with cited filenames, page numbers, and evidence chunks (`RagService`, `RagAnswer`, `Citation`, `ask_document`).
- **CLI & Integration Validation**: Flexible command-line interface supporting both mock and local Ollama generation, with automated tests validating offline mocking and optional real-service execution.

## Requirements

- Python **3.10+** (tested on 3.10–3.13)
- Runtime dependencies: `pypdf>=5.0.0`, `sentence-transformers>=3.0.0`
- Optional local LLM runtime: [Ollama](https://ollama.com) (free, open source)

## Quick Start

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# 2. Install in editable mode with dev tools
pip install -e ".[dev]"

# 3. Copy env template and set configuration
copy .env.example .env   # Windows
# cp .env.example .env   # macOS / Linux

# 4. Launch the interactive Web UI:
python -m knowledgepilot.app --serve
# Access at: http://127.0.0.1:8000

# 5. Or query a local document via CLI:
python -m knowledgepilot.app --doc sample.txt --query "What is X?" --generator ollama
```

## Running with a Real Local LLM (Ollama)

KnowledgePilot AI integrates with Ollama for 100% local, private, and free LLM generation. No API keys or paid credits are required.

### 1. Install Ollama
Download and install Ollama for your operating system:
- **Windows / macOS / Linux**: [https://ollama.com/download](https://ollama.com/download)

### 2. Pull a Local Model
Pull a lightweight model such as `llama3.2:1b` (recommended, ~1.3 GB) or `phi3`:
```bash
ollama pull llama3.2:1b
```

### 3. Ensure the Ollama Service Is Running
On Windows and macOS, the Ollama desktop app runs in the background automatically. On Linux or headless environments:
```bash
ollama serve
```
By default, the service listens at `http://localhost:11434`.

### 4. Query Documents with Local LLM Generation
Use the `--generator ollama` flag to answer questions using your local model:
```bash
# Query a PDF using real sentence-transformer embeddings and local Ollama generation:
python -m knowledgepilot.app --doc whitepaper.pdf --query "What is the key conclusion?" --generator ollama

# Custom model and endpoint:
python -m knowledgepilot.app --doc guide.txt --query "How does indexing work?" --generator ollama --ollama-model llama3.2:1b --ollama-url http://localhost:11434

# Fast verification using mock embeddings + Ollama LLM:
python -m knowledgepilot.app --doc guide.txt --query "What is discussed?" --generator ollama --mock-embeddings
```

## Python API Usage

```python
from knowledgepilot import ask_document

# 1. Ask a question using the default MockGenerator (offline, deterministic)
result = ask_document(
    doc_path="whitepaper.pdf",
    query="How does vector retrieval work?",
    top_k=3,
)

print(result.answer)
print("Citations:")
for citation in result.citations:
    print(f"- {citation.file_name} (Page {citation.page_number})")

# 2. Ask a question using a local Ollama LLM (free, local)
local_result = ask_document(
    doc_path="whitepaper.pdf",
    query="Summarize the primary architectural trade-offs.",
    top_k=3,
    generator_provider="ollama",
    ollama_model="llama3.2:1b",
    ollama_base_url="http://localhost:11434",
)

print(local_result.answer)
print(f"Sufficient evidence: {local_result.has_sufficient_evidence}")
```

## Configuration

All configuration is loaded from environment variables via `knowledgepilot.config.Settings`. See `.env.example` for the full template.

| Variable | Default | Description |
|---|---|---|
| `KNOWLEDGEPILOT_ENV` | `development` | Runtime environment (`development`, `staging`, `production`, `test`) |
| `KNOWLEDGEPILOT_LOG_LEVEL` | `INFO` | Log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) |
| `KNOWLEDGEPILOT_HOST` | `127.0.0.1` | Bind host for future API server |
| `KNOWLEDGEPILOT_PORT` | `8000` | Bind port for future API server |
| `KNOWLEDGEPILOT_GENERATOR_PROVIDER` | `mock` | Generation backend (`mock` or `ollama`) |
| `KNOWLEDGEPILOT_OLLAMA_BASE_URL` | `http://localhost:11434` | HTTP endpoint for local Ollama daemon |
| `KNOWLEDGEPILOT_OLLAMA_MODEL` | `llama3.2:1b` | Default Ollama model name |
| `OPENAI_API_KEY` | *(none)* | Optional LLM provider key — **never committed**, read only from env |
| `KNOWLEDGEPILOT_OPENAI_API_KEY` | *(fallback)* | Alternative name for the same key |

Secrets are **never** stored in source control. `Settings.from_env()` raises no exception if keys are absent — features requiring them will fail gracefully at runtime with a clear error.

## Project Structure

```
RAG_PROJECT/
├── pyproject.toml              # build, deps, ruff, mypy, pytest config
├── README.md
├── .env.example                # env template (no real secrets)
├── .gitignore
├── src/
│   └── knowledgepilot/
│       ├── __init__.py         # top-level exports including ask_document
│       ├── app.py              # application factory and CLI query tool
│       ├── config.py           # env-based configuration
│       ├── embeddings/         # Phase 3 embedding providers
│       │   ├── __init__.py
│       │   ├── base.py         # BaseEmbeddingModel interface
│       │   ├── mock.py         # MockEmbeddingModel (deterministic, test-friendly)
│       │   └── sentence_transformer.py # SentenceTransformerEmbedding (lazy load)
│       ├── generation/         # Phase 4 & 6 answer generation providers
│       │   ├── __init__.py
│       │   ├── base.py         # BaseGenerator interface
│       │   ├── errors.py       # GenerationError
│       │   ├── factory.py      # create_generator factory
│       │   ├── mock.py         # MockGenerator (deterministic, test-friendly)
│       │   ├── ollama.py       # OllamaGenerator (local LLM via HTTP REST API)
│       │   └── prompts.py      # Grounded QA prompt formatting
│       ├── ingestion/          # Phase 2 document ingestion & chunking
│       │   ├── __init__.py
│       │   ├── chunker.py      # sliding-window TextChunker
│       │   ├── errors.py       # custom ingestion exception hierarchy
│       │   ├── loaders.py      # TxtLoader, PdfLoader, load_document
│       │   └── models.py       # Document, Chunk, DocumentMetadata, ChunkMetadata
│       ├── rag/                # Phase 4 & 5 RAG orchestration
│       │   ├── __init__.py
│       │   ├── models.py       # RagAnswer, Citation
│       │   └── service.py      # RagService, ask_document
│       ├── retrieval/          # Phase 3 vector index & semantic retrieval
│       │   ├── __init__.py
│       │   ├── index.py        # VectorIndex with cosine similarity & tie-breaking
│       │   ├── models.py       # SearchResult with source & page metadata accessors
│       │   └── retriever.py    # SemanticRetriever
│       └── web/                # Phase 7 Web UI & REST API
│           ├── __init__.py
│           ├── app.py          # FastAPI application factory and endpoints
│           ├── state.py        # Thread-safe KnowledgeStore state manager
│           └── static/         # Frontend SPA assets
│               ├── app.js      # Client controller & citation interactions
│               ├── index.html  # Modern semantic glassmorphic interface
│               └── style.css   # Obsidian glassmorphic design system
└── tests/
    ├── __init__.py
    ├── test_app.py             # CLI and application factory tests
    ├── test_config.py          # Settings and environment tests
    ├── test_ingestion.py       # Ingestion and chunking tests
    ├── test_integration.py     # End-to-end integration & CLI tests
    ├── test_ollama.py          # Local LLM generation and mocked HTTP tests
    ├── test_rag.py             # Grounded QA and RAG service tests
    ├── test_retrieval.py       # Embedding, indexing, and retrieval tests
    └── test_web.py             # Web API endpoints, upload, chat, and UI tests
```

## Development & Testing

```bash
# 1. Lint checks
ruff check src tests

# 2. Code format checks
ruff format --check src tests

# 3. Static type checking (strict mode)
mypy src tests

# 4. Run standard offline deterministic tests (default, no downloads or network)
pytest

# 5. Run real model and integration tests
pytest -m integration

# 6. Run full test suite with coverage
pytest --cov=knowledgepilot --cov-report=term-missing
```

## Evidence Grounding & Citation Guarantees

KnowledgePilot AI enforces strict factual grounding:
1. **Zero Hallucination Policy**: Prompts instruct the generator to answer **only** from the retrieved evidence chunks.
2. **Explicit Uncertainty**: When retrieved context is empty or lacks sufficient facts, the system returns:
   > *"I do not have sufficient information in the provided documents to answer this question."*
3. **Citation Integrity**: Citations link strictly to actual source files and 1-based page numbers parsed from documents. Citations are never fabricated.

## Roadmap

- **Phase 1** — Project foundation (completed)
- **Phase 2** — Document ingestion & chunking (completed)
- **Phase 3** — Local embeddings & semantic retrieval (completed)
- **Phase 4** — Question answering grounded in retrieved documents (completed)
- **Phase 5** — End-to-end pipeline integration and validation (completed)
- **Phase 6** — Real local LLM generation via Ollama (completed)
- **Phase 7** — Modern Web UI with document upload, grounded chat, & citation inspector (completed)
- **Phase 8** — Real-time token streaming (SSE) with interactive citation pills (completed)
- **Phase 9** — SHA-256 document deduplication & atomic disk persistence (completed)
- **Phase 10** — Hybrid search (BM25 + Dense Vectors + Reciprocal Rank Fusion) (completed)
- **Phase 11** — Automated RAG evaluation harness (Recall@K, MRR, Refusal Rate) (completed)
- **Phase 12** — Multi-stage Docker containerization & GitHub Actions CI matrix (completed)

## License

MIT
