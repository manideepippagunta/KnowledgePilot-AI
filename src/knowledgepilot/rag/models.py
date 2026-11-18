"""Data models for grounded question answering and RAG responses."""

from __future__ import annotations

from dataclasses import dataclass

from knowledgepilot.retrieval.models import SearchResult


@dataclass(frozen=True, slots=True)
class Citation:
    """Document citation indicating the source and page of supporting evidence.

    Attributes:
        file_name: Original filename of the supporting document.
        page_number: 1-based page number if available (e.g. for PDFs), else None.
        source: Full source path or URI.
        chunk_index: Index of the specific supporting chunk.
        document_index: 1-based index corresponding to [Document X] in prompt.
        snippet: Concise preview snippet of supporting passage text.
    """

    file_name: str
    page_number: int | None
    source: str
    chunk_index: int
    document_index: int = 1
    snippet: str = ""


@dataclass(frozen=True, slots=True)
class RagAnswer:
    """Structured response containing generated answer and supporting evidence.

    Attributes:
        query: The user's input question.
        answer: The generated answer text.
        citations: Sequence of Citations corresponding to retrieved evidence.
        evidence: Raw retrieved SearchResult instances used during generation.
        has_sufficient_evidence: True if evidence was present and supported an answer;
            False if the system recognized insufficient context.
    """

    query: str
    answer: str
    citations: list[Citation]
    evidence: list[SearchResult]
    has_sufficient_evidence: bool = True
    provider: str = "mock"
    model: str = "mock"

    @property
    def cited_files(self) -> list[str]:
        """Unique list of cited filenames preserving order of appearance."""
        seen: set[str] = set()
        out: list[str] = []
        for c in self.citations:
            if c.file_name not in seen:
                seen.add(c.file_name)
                out.append(c.file_name)
        return out

    @property
    def cited_pages(self) -> list[tuple[str, int | None]]:
        """Unique list of (filename, page_number) tuples preserving order."""
        seen: set[tuple[str, int | None]] = set()
        out: list[tuple[str, int | None]] = []
        for c in self.citations:
            item = (c.file_name, c.page_number)
            if item not in seen:
                seen.add(item)
                out.append(item)
        return out

    @property
    def directly_cited_documents(self) -> list[Citation]:
        """Return citations explicitly referenced by [Document X] in the answer text."""
        import re

        pattern = re.compile(r"\[(?:Document|Doc)\s*(\d+)", re.IGNORECASE)
        matches = {int(m) for m in pattern.findall(self.answer)}
        return [c for c in self.citations if c.document_index in matches]

    @property
    def directly_cited_indices(self) -> list[int]:
        """Return sorted unique 1-based doc indices explicitly cited in answer text."""
        return sorted({c.document_index for c in self.directly_cited_documents})
