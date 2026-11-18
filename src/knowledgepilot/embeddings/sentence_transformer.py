"""Local sentence-transformer embedding model implementation with lazy loading."""

from __future__ import annotations

from typing import Any

from knowledgepilot.embeddings.base import BaseEmbeddingModel

_KNOWN_DIMENSIONS: dict[str, int] = {
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "paraphrase-MiniLM-L6-v2": 384,
}


class SentenceTransformerEmbedding(BaseEmbeddingModel):
    """Local open-source embedding model using sentence-transformers.

    Model loading is deferred until first inference to keep imports fast and
    avoid unnecessary resource consumption during testing.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
    ) -> None:
        """Initialize the sentence transformer embedding wrapper.

        Args:
            model_name: HuggingFace model identifier.
            device: Optional compute device (e.g. 'cpu', 'cuda').
        """
        self.model_name = model_name
        self.device = device
        self._model: Any = None
        self._dimension: int | None = _KNOWN_DIMENSIONS.get(model_name)

    def _get_model(self) -> Any:
        """Lazily load and cache the SentenceTransformer model instance."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for "
                    "SentenceTransformerEmbedding. "
                    "Install it via `pip install sentence-transformers`."
                ) from exc

            self._model = SentenceTransformer(self.model_name, device=self.device)
            if self._dimension is None:
                dim = self._model.get_sentence_embedding_dimension()
                if isinstance(dim, int):
                    self._dimension = dim
        return self._model

    @property
    def dimension(self) -> int:
        """Dimensionality of the embedding vectors produced by this model."""
        if self._dimension is not None:
            return self._dimension
        self._get_model()
        assert self._dimension is not None
        return self._dimension

    def embed_text(self, text: str) -> list[float]:
        """Embed a single piece of text into a vector.

        Args:
            text: Input string to embed.

        Returns:
            A list of floats representing the embedding vector.

        Raises:
            ValueError: If the text is empty or whitespace-only.
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty or whitespace-only text.")
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple pieces of text in batches.

        Args:
            texts: List of strings to embed.

        Returns:
            A list of float vectors, one per input string.

        Raises:
            ValueError: If any text in the list is empty or whitespace-only.
        """
        if not texts:
            return []

        for idx, text in enumerate(texts):
            if not text or not text.strip():
                raise ValueError(f"Text at index {idx} is empty or whitespace-only.")

        model = self._get_model()
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return [[float(x) for x in vec] for vec in embeddings]
