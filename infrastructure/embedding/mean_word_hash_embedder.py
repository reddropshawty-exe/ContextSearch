"""Эмбеддер, усредняющий хэшированные векторы слов (упрощённый GloVe)."""
from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Sequence

from domain.entities import Query
from domain.interfaces import Embedder


class MeanWordHashEmbedder(Embedder):
    """Создаёт детерминированные векторы, хэшируя отдельные слова."""

    def __init__(self, dimension: int = 32) -> None:
        self._dimension = dimension
        self._model_id = f"hash-mean-word-{dimension}"

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimension(self) -> int:
        return self._dimension

    def _word_vector(self, word: str) -> list[float]:
        digest = hashlib.md5(word.encode("utf-8")).digest()
        vector = [digest[i % len(digest)] / 255.0 for i in range(self._dimension)]
        return vector

    def _combine(self, words: Sequence[str]) -> list[float]:
        if not words:
            return [0.0] * self._dimension
        counts = Counter(word.lower() for word in words if word.strip())
        vector = [0.0] * self._dimension
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
