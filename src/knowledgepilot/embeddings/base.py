"""Provider-independent base interface for embedding models."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbeddingModel(ABC):
    """Abstract base class for text embedding models.

    Implementations can be local (e.g. sentence-transformers) or remote APIs.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """The dimensionality of the embedding vectors produced by this model."""

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Embed a single piece of text into a vector.

        Args:
            text: Input string to embed.

        Returns:
            A list of floats representing the embedding vector.

        Raises:
            ValueError: If the text is empty or whitespace-only.
        """

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple pieces of text in batches.

        Args:
            texts: List of strings to embed.

        Returns:
            A list of float vectors, one per input string.

        Raises:
            ValueError: If any text in the list is empty or whitespace-only.
        """
