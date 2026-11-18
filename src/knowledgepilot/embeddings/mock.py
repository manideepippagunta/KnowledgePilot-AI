"""Deterministic mock embedding model for testing and offline development."""

from __future__ import annotations

import math
from collections.abc import Mapping

from knowledgepilot.embeddings.base import BaseEmbeddingModel


def _normalize(vec: list[float]) -> list[float]:
    """Normalize a vector to unit length (L2 norm)."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


class MockEmbeddingModel(BaseEmbeddingModel):
    """Deterministic, lightweight embedding model without external dependencies.

    Produces predictable, unit-normalized float vectors for testing semantic
    retrieval and vector indexing logic without downloading large neural models.
    """

    def __init__(
        self,
        dimension: int = 4,
        vectors: Mapping[str, list[float]] | None = None,
    ) -> None:
        """Initialize the mock embedding model.

        Args:
            dimension: Dimensionality of output vectors. Must be > 0.
            vectors: Optional fixed dictionary mapping specific text strings
                to predetermined embedding vectors.

        Raises:
            ValueError: If dimension <= 0, or if any custom vector length does
                not match dimension.
        """
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")

        self._dimension = dimension
        self._preset_vectors: dict[str, list[float]] = {}

        if vectors is not None:
            for text_key, vec in vectors.items():
                if len(vec) != dimension:
                    raise ValueError(
                        f"Custom vector for {text_key!r} has length {len(vec)}, "
                        f"expected {dimension}."
                    )
                self._preset_vectors[text_key] = _normalize(vec)

    @property
    def dimension(self) -> int:
        return self._dimension

    def _generate_vector(self, text: str) -> list[float]:
        """Generate a deterministic unit-normalized vector for the given string."""
        if text in self._preset_vectors:
            return list(self._preset_vectors[text])

        # Deterministic hashing into dimension buckets
        raw = [0.0] * self._dimension
        for i, char in enumerate(text):
            raw[(ord(char) + i) % self._dimension] += 1.0

        return _normalize(raw)

    def embed_text(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("Cannot embed empty or whitespace-only text.")
        return self._generate_vector(text.strip())

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        for idx, text in enumerate(texts):
            if not text or not text.strip():
                raise ValueError(f"Text at index {idx} is empty or whitespace-only.")

        return [self._generate_vector(t.strip()) for t in texts]
