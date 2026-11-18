"""Provider-independent interface for answer generation models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any


class BaseGenerator(ABC):
    """Abstract base class for answer and text generation providers."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a text answer for the provided prompt.

        Args:
            prompt: Formatted prompt containing instructions and retrieved context.
            **kwargs: Optional model generation arguments.

        Returns:
            The generated text response.

        Raises:
            GenerationError: If the model or backend fails.
            ValueError: If the prompt is empty or invalid.
        """

    def generate_stream(self, prompt: str) -> Iterator[str]:
        """Stream generated tokens for the given prompt.

        Default implementation yields the full response from ``generate(prompt)``
        as a single chunk so subclasses remain backwards-compatible.

        Args:
            prompt: Formatted prompt containing instructions and retrieved context.

        Yields:
            Individual token strings as they are produced.
        """
        yield self.generate(prompt)
