"""Prompt templates and formatters for grounded question answering."""

from __future__ import annotations

from collections.abc import Sequence

from knowledgepilot.retrieval.models import SearchResult

DEFAULT_SYSTEM_INSTRUCTION = (
    "You are KnowledgePilot AI, an evidence-grounded question-answering assistant.\n"
    "Answer the user's question using ONLY the directly stated facts in the provided "
    "EVIDENCE passages below.\n\n"
    "Strict Grounding Rules:\n"
    "1. Use ONLY facts directly and explicitly stated in the provided evidence. Do NOT "
    "extrapolate, speculate, assume, or incorporate outside knowledge about "
    "companies, systems, architectures, or technical workflows (even if factually "
    "true in reality).\n"
    "2. Cite every factual claim, step, or phase with its source tag in brackets, for "
    "example: [Document 1, Page 2] or [Document 1] if no page number is available.\n"
    "3. Distinguish explicitly stated facts from missing information:\n"
    "   - For workflows, pipelines, multi-phase systems, or multi-part questions, "
    "describe ONLY the specific phases, steps, or components that are explicitly "
    "explained.\n"
    "   - If a phase or component is named but the evidence provides no details on how "
    "it operates, explicitly state that the provided documents do not describe its "
    "implementation or details. NEVER invent or fabricate phase descriptions, "
    "mechanisms, metrics, or technologies.\n"
    "   - For multi-part queries where some parts are addressed and others are "
    "omitted, answer the supported parts with citations and explicitly specify which "
    "parts are missing from the documents.\n"
    "4. If the provided evidence contains NO relevant facts to answer the question, "
    "state exactly: 'I do not have sufficient information in the provided documents "
    "to answer this question.'"
)


def format_grounded_qa_prompt(
    query: str,
    evidence: Sequence[SearchResult],
    system_instruction: str | None = None,
) -> str:
    """Format a grounded question-answering prompt with retrieved evidence.

    Args:
        query: User question to answer.
        evidence: Sequence of retrieved SearchResult items.
        system_instruction: Optional custom system instruction.

    Returns:
        The formatted prompt string ready for LLM consumption.

    Raises:
        ValueError: If query is empty or whitespace-only.
    """
    if not query or not query.strip():
        raise ValueError("Query string cannot be empty or whitespace-only.")

    system_text = (
        system_instruction.strip()
        if system_instruction is not None and system_instruction.strip()
        else DEFAULT_SYSTEM_INSTRUCTION
    )

    evidence_blocks: list[str] = []
    for idx, item in enumerate(evidence, start=1):
        page_str = (
            f"Page: {item.page_number}" if item.page_number is not None else "Page: N/A"
        )
        header = f"[Document {idx}] (File: {item.file_name}, {page_str})"
        evidence_blocks.append(f"{header}\n{item.text}")

    context_str = (
        "\n\n".join(evidence_blocks) if evidence_blocks else "No evidence provided."
    )

    return (
        f"{system_text}\n\n"
        f"--- EVIDENCE ---\n"
        f"{context_str}\n"
        f"--- END EVIDENCE ---\n\n"
        f"Question: {query.strip()}\n"
        f"Answer:"
    )
