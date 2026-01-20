"""Embedder that uses hashed character n-grams (FastText-like toy)."""
from __future__ import annotations

import hashlib
import math
from typing import Sequence

from domain.entities import Query
from domain.interfaces import Embedder


class CharacterNgramEmbedder(Embedder):
    """Simple char n-gram embedder that supports multiple n sizes."""

    def __init__(self, dimension: int = 24, ngram_sizes: Sequence[int] | None = None) -> None:
        self._dimension = dimension
        self._model_id = f"hash-char-ngram-{dimension}"
        self.ngram_sizes = tuple(ngram_sizes or (3, 4, 5))

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimension(self) -> int:
        return self._dimension

    def _ngrams(self, text: str) -> list[str]:
        clean = text.replace("\n", " ")
        grams: list[str] = []
        for n in self.ngram_sizes:
            if n <= 0:
                continue
            grams.extend(
                clean[i : i + n]
                for i in range(max(len(clean) - n + 1, 0))
                if clean[i : i + n].strip()
            )
        return grams or [clean]

    def _vectorize(self, text: str) -> list[float]:
        grams = self._ngrams(text)
        vector = [0.0] * self._dimension
        for gram in grams:
            digest = hashlib.sha1(gram.encode("utf-8")).digest()
            for idx in range(self._dimension):
                vector[idx] += digest[idx % len(digest)] / 255.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vectorize(text) for text in texts]

    def embed_query(self, query: Query) -> list[float]:
        return self._vectorize(query.text)


__all__ = ["CharacterNgramEmbedder"]
