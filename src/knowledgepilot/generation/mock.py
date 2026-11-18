"""Deterministic mock answer generator for testing and offline development."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Any

from knowledgepilot.generation.base import BaseGenerator
from knowledgepilot.generation.errors import GenerationError


class MockGenerator(BaseGenerator):
    """Deterministic, zero-dependency answer generator for unit testing.

    Allows tests to simulate successful grounded answers, insufficient evidence
    disclaimers, or model failures without external API calls or weight downloads.
    """

    def __init__(
        self,
        default_response: str = (
            "Based on the provided documents, the answer is verified."
        ),
        responses: Mapping[str, str] | None = None,
        custom_fn: Callable[[str], str] | None = None,
        should_fail: bool = False,
        error_message: str = "Simulated generation error",
    ) -> None:
        """Initialize the mock generator.

        Args:
            default_response: Fallback response if no specific match is found.
            responses: Optional mapping of keyword substrings to return values.
            custom_fn: Optional callable transforming prompt into answer.
            should_fail: If True, generate() raises GenerationError.
            error_message: Error message when should_fail is True.
        """
        self.default_response = default_response
        self.responses = dict(responses) if responses is not None else {}
        self.custom_fn = custom_fn
        self.should_fail = should_fail
        self.error_message = error_message
        self.calls: list[str] = []

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response according to configuration."""
        _ = kwargs
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty or whitespace-only.")

        self.calls.append(prompt)

        if self.should_fail:
            raise GenerationError(self.error_message)

        if self.custom_fn is not None:
            return self.custom_fn(prompt)

        for key, resp in self.responses.items():
            if key in prompt:
                return resp

        return self.default_response

    def generate_stream(self, prompt: str) -> Iterator[str]:
        """Deterministically stream the mock response word-by-word.

        Splits the full ``generate()`` output on spaces so unit tests and the
        UI can exercise the streaming code path without any network calls.

        Args:
            prompt: Formatted prompt (passed unchanged to ``generate``).

        Yields:
            Individual words, each trailed by a space except the last.
        """
        full_text: str = self.generate(prompt)
        words = full_text.split(" ")
        for idx, word in enumerate(words):
            yield word if idx == len(words) - 1 else f"{word} "
