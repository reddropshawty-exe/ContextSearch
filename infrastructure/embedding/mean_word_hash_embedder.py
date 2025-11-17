"""Embedder that averages hashed word vectors (GloVe-like toy model)."""
from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Sequence

from domain.entities import Query
from domain.interfaces import Embedder


class MeanWordHashEmbedder(Embedder):
    """Produces deterministic vectors by hashing individual words."""

    def __init__(self, dimension: int = 32) -> None:
        self.dimension = dimension

    def _word_vector(self, word: str) -> list[float]:
        digest = hashlib.md5(word.encode("utf-8")).digest()
        vector = [digest[i % len(digest)] / 255.0 for i in range(self.dimension)]
        return vector

    def _combine(self, words: Sequence[str]) -> list[float]:
        if not words:
            return [0.0] * self.dimension
        counts = Counter(word.lower() for word in words if word.strip())
        vector = [0.0] * self.dimension
        total = sum(counts.values()) or 1
        for word, count in counts.items():
            word_vec = self._word_vector(word)
            for idx, value in enumerate(word_vec):
                vector[idx] += value * count / total
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._combine(text.split()) for text in texts]

    def embed_query(self, query: Query) -> list[float]:
        return self._combine(query.text.split())


__all__ = ["MeanWordHashEmbedder"]
