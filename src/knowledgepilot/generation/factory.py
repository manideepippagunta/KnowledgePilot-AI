"""Generator factory for selecting mock or local LLM backends."""

from __future__ import annotations

from knowledgepilot.generation.base import BaseGenerator
from knowledgepilot.generation.mock import MockGenerator
from knowledgepilot.generation.ollama import OllamaGenerator


def create_generator(
    provider: str = "mock",
    *,
    base_url: str | None = None,
    model: str | None = None,
) -> BaseGenerator:
    """Instantiate an answer generator by provider name.

    Args:
        provider: Provider identifier ('mock' or 'ollama').
        base_url: Optional base URL override (for 'ollama').
        model: Optional model name override (for 'ollama').

    Returns:
        Configured BaseGenerator instance.

    Raises:
        ValueError: If the provider is unsupported.
    """
    clean_provider = provider.lower().strip()
    if clean_provider == "mock":
        return MockGenerator()
    if clean_provider == "ollama":
        resolved_base_url = (
            base_url.rstrip("/")
            if base_url is not None and base_url.strip()
            else "http://localhost:11434"
        )
        resolved_model = (
            model.strip() if model is not None and model.strip() else "llama3.2:3b"
        )
        return OllamaGenerator(base_url=resolved_base_url, model=resolved_model)

    raise ValueError(
        f"Unsupported generator provider '{provider}'. "
        "Supported providers: 'mock', 'ollama'."
    )
