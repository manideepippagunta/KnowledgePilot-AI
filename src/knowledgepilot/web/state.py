"""State management and in-memory knowledge store for the KnowledgePilot Web API."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import threading
import uuid
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from knowledgepilot.embeddings.base import BaseEmbeddingModel
from knowledgepilot.embeddings.sentence_transformer import (
    SentenceTransformerEmbedding,
)
from knowledgepilot.generation.base import BaseGenerator
from knowledgepilot.generation.factory import create_generator
from knowledgepilot.generation.mock import MockGenerator
from knowledgepilot.ingestion.chunker import TextChunker
from knowledgepilot.ingestion.loaders import load_document
from knowledgepilot.ingestion.models import Chunk, Document, DocumentMetadata
from knowledgepilot.rag.models import RagAnswer
from knowledgepilot.rag.service import RagService
from knowledgepilot.retrieval.index import (
    VectorIndex,
    vector_index_export_entries,
    vector_index_import_entries,
)
from knowledgepilot.retrieval.retriever import SemanticRetriever

logger = logging.getLogger(__name__)


def compute_sha256(content: bytes) -> str:
    """Compute a deterministic lowercase hex SHA-256 digest for raw file bytes."""
    return hashlib.sha256(content).hexdigest()


def sanitize_filename(file_name: str) -> str:
    """Sanitize user-provided filename to a safe basename for Windows and POSIX.

    Strips directory components, drive letters, invalid characters
    (< > : \" / \\ | ? *), control characters, and trailing dots/spaces.
    """
    if not file_name:
        return "document"

    # Normalize slashes and strip trailing dots/spaces
    base_name = file_name.replace("\\", "/").split("/")[-1].strip()
    base_name = base_name.rstrip(". ")
    if not base_name:
        return "document"

    if "." in base_name:
        last_dot = base_name.rfind(".")
        raw_stem = base_name[:last_dot]
        ext = base_name[last_dot:].lower().rstrip(". ")
    else:
        raw_stem = base_name
        ext = ""

    # Strip characters illegal on Windows NTFS and ASCII control chars
    clean_stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", raw_stem).strip(". ")
    if not clean_stem:
        clean_stem = "document"

    return f"{clean_stem}{ext}"


@dataclass(frozen=True, slots=True)
class DocumentSummary:
    """Summary of an indexed document in the web knowledge store."""

    doc_id: str
    file_name: str
    file_type: str
    file_size_bytes: int
    page_count: int
    chunk_count: int
    uploaded_at: str


class PersistentKnowledgeStoreMixin:
    """Drop-in persistence & SHA-256 deduplication methods for `KnowledgeStore`.

    Features:
    - **SHA-256 Deduplication**: Maps `content_hash -> DocumentSummary`. If an
      uploaded file's SHA-256 matches an already-indexed document, returns the
      existing `DocumentSummary` immediately without re-chunking or re-embedding.
    - **Atomic JSON Persistence**: When `storage_path` is set (e.g.
      `.knowledgepilot/index_state.json`), writes documents, hashes, and vector
      index entries to a temporary file and atomically replaces the target file
      via `os.replace()`.
    - **Backward-Compatible**: When `storage_path=None` (default in unit tests),
      behaves purely in-memory so all existing tests remain isolated.
    """

    _storage_path: Path | None
    _hash_to_doc_id: dict[str, str]
    _documents: dict[str, Any]
    _doc_counter: int

    def init_persistence(self, storage_path: str | Path | None = None) -> None:
        """Initialize hash tracking and load persisted state if path exists."""
        self._hash_to_doc_id = {}
        self._storage_path = Path(storage_path) if storage_path else None
        if self._storage_path is not None and self._storage_path.exists():
            self._load_from_disk()

    def find_duplicate_by_bytes(
        self, content: bytes
    ) -> tuple[str, DocumentSummary | None]:
        """Return `(sha256_hex, existing_doc_summary_or_None)`."""
        digest = compute_sha256(content)
        existing_doc_id = self._hash_to_doc_id.get(digest)
        docs: dict[str, DocumentSummary] = getattr(self, "documents", {})
        if existing_doc_id and existing_doc_id in docs:
            return digest, docs[existing_doc_id]
        return digest, None

    def register_document_hash(self, content_hash: str, doc_id: str) -> None:
        """Record a document's SHA-256 digest after successful indexing."""
        self._hash_to_doc_id[content_hash] = doc_id

    def save_to_disk(self) -> None:
        """Atomically persist documents, SHA-256 map, and vector index entries."""
        if self._storage_path is None:
            return

        self._storage_path.parent.mkdir(parents=True, exist_ok=True)

        docs: dict[str, DocumentSummary] = getattr(self, "documents", {})
        docs_payload: list[dict[str, Any]] = []
        for doc in docs.values():
            dump_fn = getattr(doc, "model_dump", None)
            if callable(dump_fn):
                doc_dict = dump_fn()
            elif is_dataclass(doc) and not isinstance(doc, type):
                doc_dict = asdict(doc)
            else:
                doc_dict = dict(doc.__dict__)
            docs_payload.append(doc_dict)

        index = getattr(self, "_index", None)
        index_entries: list[dict[str, Any]]
        if index is not None and hasattr(index, "to_list"):
            index_entries = index.to_list()
        elif index is not None:
            index_entries = vector_index_export_entries(index)
        else:
            index_entries = []

        state_payload = {
            "version": 1,
            "documents": docs_payload,
            "hash_to_doc_id": dict(self._hash_to_doc_id),
            "index_entries": index_entries,
        }

        fd, tmp_path_str = tempfile.mkstemp(
            dir=str(self._storage_path.parent),
            prefix=".kp_state_",
            suffix=".tmp",
        )
        tmp_path = Path(tmp_path_str)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(state_payload, handle, ensure_ascii=False)
            os.replace(tmp_path, self._storage_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise

    def _load_from_disk(
        self,
        doc_summary_cls: type[Any] | None = None,
        chunk_cls: type[Any] | None = None,
    ) -> None:
        """Restore persisted documents, SHA-256 map, and vector index entries."""
        if self._storage_path is None or not self._storage_path.exists():
            return

        try:
            raw_text = self._storage_path.read_text(encoding="utf-8")
            if not raw_text.strip():
                return
            data = json.loads(raw_text)
        except Exception as exc:
            logger.warning(
                "Could not load persisted index state from %s: %s",
                self._storage_path,
                exc,
            )
            return

        self._hash_to_doc_id = {
            str(k): str(v) for k, v in data.get("hash_to_doc_id", {}).items()
        }

        doc_cls = doc_summary_cls or DocumentSummary
        restored_docs: dict[str, DocumentSummary] = {}
        for raw_doc in data.get("documents", []):
            doc_obj = doc_cls(**raw_doc)
            restored_docs[doc_obj.doc_id] = doc_obj

        if hasattr(self, "documents"):
            self.documents = restored_docs
        self._documents = restored_docs
        self._doc_counter = len(restored_docs)

        chunk_klass = chunk_cls or Chunk
        index = getattr(self, "_index", None)
        if index is not None:
            if hasattr(index, "load_entries"):
                index.load_entries(data.get("index_entries", []), chunk_cls=chunk_klass)
            else:
                vector_index_import_entries(
                    index, data.get("index_entries", []), chunk_cls=chunk_klass
                )

    def clear_persisted_disk_state(self) -> None:
        """Clear hash map and remove persisted state file on `clear()`."""
        self._hash_to_doc_id.clear()
        if self._storage_path is not None and self._storage_path.exists():
            self._storage_path.unlink(missing_ok=True)


@dataclass
class KnowledgeStore(PersistentKnowledgeStoreMixin):
    """Thread-safe store managing ingested documents, vector index, and RAG service."""

    embedding_model: BaseEmbeddingModel = field(
        default_factory=SentenceTransformerEmbedding
    )
    generator: BaseGenerator = field(default_factory=MockGenerator)
    chunker: TextChunker = field(
        default_factory=lambda: TextChunker(chunk_size=500, chunk_overlap=50)
    )
    documents: dict[str, DocumentSummary] = field(default_factory=dict)
    storage_path: Path | str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _index: VectorIndex = field(default_factory=VectorIndex)
    _retriever: SemanticRetriever | None = None
    _doc_counter: int = 0

    def __post_init__(self) -> None:
        self.init_persistence(self.storage_path)
        self._retriever = SemanticRetriever(
            embedding_model=self.embedding_model,
            index=self._index,
        )

    @property
    def total_chunks(self) -> int:
        """Total number of indexed vector chunks."""
        with self._lock:
            return self._index.total_chunks

    @property
    def total_documents(self) -> int:
        """Total number of uploaded documents."""
        with self._lock:
            return len(self.documents)

    def list_documents(self) -> list[DocumentSummary]:
        """Return list of all indexed document summaries."""
        with self._lock:
            return list(self.documents.values())

    def add_document(self, file_name: str, data: bytes) -> DocumentSummary:
        """Ingest, chunk, and index a new document into the knowledge store.

        Args:
            file_name: Original filename (e.g. 'report.pdf' or 'notes.txt').
            data: Raw binary content of the file.

        Returns:
            DocumentSummary with statistics on the indexed document.

        Raises:
            ValueError: If file content is empty or file format is unsupported.
        """
        if not data:
            raise ValueError(f"Uploaded file '{file_name}' is empty.")

        clean_name = sanitize_filename(file_name)
        ext = Path(clean_name).suffix.lower()
        if ext not in {".txt", ".pdf"}:
            raise ValueError(
                f"Unsupported file format '{ext}' for '{file_name}'. "
                "Supported formats: .txt, .pdf"
            )

        with self._lock:
            digest, duplicate = self.find_duplicate_by_bytes(data)
            if duplicate is not None:
                logger.info(
                    "Duplicate document detected for '%s' (hash=%s, existing_id=%s). "
                    "Skipping re-indexing.",
                    clean_name,
                    digest,
                    duplicate.doc_id,
                )
                return duplicate

        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_file_name = f"upload_{uuid.uuid4().hex[:12]}{ext}"
            temp_file = Path(tmp_dir) / temp_file_name
            temp_file.write_bytes(data)

            loaded_docs = load_document(temp_file)
            page_count = len(loaded_docs)

            # Preserve the clean source filename and PDF page numbers in citations
            normalized_docs = [
                Document(
                    text=doc.text,
                    metadata=DocumentMetadata(
                        source=clean_name,
                        file_name=clean_name,
                        file_type=ext.lstrip("."),
                        page_number=doc.metadata.page_number,
                        total_pages=doc.metadata.total_pages,
                    ),
                )
                for doc in loaded_docs
            ]

            chunks = self.chunker.split_documents(normalized_docs)
            if not chunks:
                raise ValueError(
                    f"Document '{clean_name}' contains no extractable text chunks."
                )

            # Pre-compute embeddings before locking and mutating index state
            texts = [c.text for c in chunks]
            embeddings = self.embedding_model.embed_texts(texts)

            with self._lock:
                digest, duplicate = self.find_duplicate_by_bytes(data)
                if duplicate is not None:
                    return duplicate

                assert self._retriever is not None
                initial_entries = list(self._index._entries)
                try:
                    self._index.add_batch(chunks, embeddings)
                    self._doc_counter += 1
                    ts = int(datetime.now(timezone.utc).timestamp())
                    doc_id = f"doc_{self._doc_counter}_{ts}"
                    summary = DocumentSummary(
                        doc_id=doc_id,
                        file_name=clean_name,
                        file_type=ext.lstrip("."),
                        file_size_bytes=len(data),
                        page_count=page_count,
                        chunk_count=len(chunks),
                        uploaded_at=datetime.now(timezone.utc).isoformat(),
                    )
                    self.documents[doc_id] = summary
                    self.register_document_hash(digest, doc_id)
                    self.save_to_disk()
                    logger.info(
                        "Indexed document: %s (%d pages, %d chunks)",
                        clean_name,
                        page_count,
                        len(chunks),
                    )
                    return summary
                except Exception:
                    # Roll back index entries on failure to preserve vector integrity
                    self._index._entries = initial_entries
                    raise

    def answer_question(
        self,
        query: str,
        top_k: int = 3,
        **kwargs: Any,
    ) -> RagAnswer:
        """Execute grounded question answering against all indexed documents.

        Args:
            query: User's question.
            top_k: Number of evidence chunks to retrieve.
            **kwargs: Extra parameters passed to generator.

        Returns:
            Structured RagAnswer with citations and evidence.
        """
        with self._lock:
            assert self._retriever is not None
            service = RagService(
                retriever=self._retriever,
                generator=self.generator,
            )
            return service.answer_question(query, top_k=top_k, **kwargs)

    def set_generator(
        self,
        provider: str,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        """Update active generator provider.

        Args:
            provider: 'mock' or 'ollama'.
            base_url: Optional Ollama endpoint.
            model: Optional Ollama model name.
        """
        with self._lock:
            self.generator = create_generator(
                provider=provider,
                base_url=base_url,
                model=model,
            )
            logger.info("Generator updated to: provider=%s model=%s", provider, model)

    def clear(self) -> None:
        """Reset the knowledge store, clearing all documents and embeddings."""
        with self._lock:
            self.documents.clear()
            self._doc_counter = 0
            self._index = VectorIndex()
            self._retriever = SemanticRetriever(
                embedding_model=self.embedding_model,
                index=self._index,
            )
            self.clear_persisted_disk_state()
            logger.info("Knowledge store cleared.")
