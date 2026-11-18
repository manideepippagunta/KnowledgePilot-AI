"""Exception hierarchy for text and answer generation."""

from __future__ import annotations


class GenerationError(Exception):
    """Raised when an answer generator encounters an error during inference."""
