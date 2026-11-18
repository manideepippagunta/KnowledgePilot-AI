"""Pure-Python Okapi BM25 sparse lexical index and Reciprocal Rank Fusion (RRF).

Zero external dependencies. Designed to complement dense vector retrieval
(`VectorIndex`) for exact keyword, acronym, and identifier matching.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

from knowledgepilot.ingestion.models import Chunk
from knowledgepilot.retrieval.models import SearchResult

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric terms."""
    return [tok.lower() for tok in _TOKEN_RE.findall(text)]


class BM25Index:
    """In-memory Okapi BM25 index over `Chunk` objects."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 <= 0.0:
            raise ValueError(f"k1 must be positive, got {k1}")
        if not 0.0 <= b <= 1.0:
            raise ValueError(f"b must be in [0.0, 1.0], got {b}")

        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._doc_freqs: list[Counter[str]] = []
        self._doc_lens: list[int] = []
        self._df: Counter[str] = Counter()
        self._avgdl: float = 0.0

    def __len__(self) -> int:
        return len(self._chunks)

    def add_batch(self, chunks: Sequence[Chunk]) -> None:
        """Index a sequence of chunks for BM25 lexical search."""
        for chunk in chunks:
            tokens = tokenize(chunk.text)
            tf = Counter(tokens)
            self._chunks.append(chunk)
            self._doc_freqs.append(tf)
            self._doc_lens.append(len(tokens))
            for term in tf:
                self._df[term] += 1

        total_len = sum(self._doc_lens)
        self._avgdl = total_len / len(self._chunks) if self._chunks else 0.0

    def clear(self) -> None:
        """Remove all indexed chunks."""
        self._chunks.clear()
        self._doc_freqs.clear()
        self._doc_lens.clear()
        self._df.clear()
        self._avgdl = 0.0

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        """Score all chunks with Okapi BM25 and return top-K results."""
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if not self._chunks:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        n_docs = len(self._chunks)
        avgdl = self._avgdl if self._avgdl > 0.0 else 1.0
        scored: list[SearchResult] = []

        for idx, chunk in enumerate(self._chunks):
            tf_map = self._doc_freqs[idx]
            dl = self._doc_lens[idx]
            score = 0.0

            for term in query_tokens:
                freq = tf_map.get(term, 0)
                if freq == 0:
                    continue
                df = self._df.get(term, 0)
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                denom = freq + self.k1 * (1.0 - self.b + self.b * (dl / avgdl))
                score += idf * ((freq * (self.k1 + 1.0)) / denom)

            if score > 0.0:
                scored.append(SearchResult(chunk=chunk, score=score))

        scored.sort(
            key=lambda r: (
                -r.score,
                r.chunk.metadata.file_name,
                r.chunk.metadata.chunk_index,
                r.chunk.text,
            )
        )
        return scored[:top_k]


def reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[SearchResult]],
    k: int = 60,
    top_k: int = 3,
) -> list[SearchResult]:
    """Fuse multiple ranked `SearchResult` lists via Reciprocal Rank Fusion (RRF).

    RRF score for document d = sum_{L} 1 / (k + rank_L(d)) where rank is 1-based.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")

    rrf_scores: dict[tuple[str, int, str], float] = {}
    chunk_by_key: dict[tuple[str, int, str], Chunk] = {}

    for results in result_lists:
        for rank_0, res in enumerate(results):
            key = (
                res.chunk.metadata.file_name,
                res.chunk.metadata.chunk_index,
                res.chunk.text,
            )
            chunk_by_key[key] = res.chunk
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (k + rank_0 + 1))

    fused = [
        SearchResult(chunk=chunk_by_key[key], score=score)
        for key, score in rrf_scores.items()
    ]
    fused.sort(
        key=lambda r: (
            -r.score,
            r.chunk.metadata.file_name,
            r.chunk.metadata.chunk_index,
            r.chunk.text,
        )
    )
    return fused[:top_k]
