"""Хранилище эмбеддингов в памяти для демо."""
from __future__ import annotations

import heapq
from typing import Sequence

from domain.entities import Chunk, RetrievalResult
from domain.interfaces import EmbeddingStore


class InMemoryEmbeddingStore(EmbeddingStore):
    """Хранит эмбеддинги в списках Python и ищет перебором."""

    def __init__(self, collection_id: str = "in-memory") -> None:
        self.collection_id = collection_id
        self._entries: list[tuple[Chunk, list[float]]] = []

    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        for chunk, embedding in zip(chunks, embeddings):
            self._entries.append((chunk, list(embedding)))

    def search(self, query_embedding: Sequence[float], top_k: int = 5) -> list[RetrievalResult]:
        scored: list[tuple[float, Chunk]] = []
        for chunk, embedding in self._entries:
            score = self._cosine_similarity(query_embedding, embedding)
            heapq.heappush(scored, (score, chunk))
            if len(scored) > top_k:
                heapq.heappop(scored)

        sorted_results = sorted(scored, key=lambda item: item[0], reverse=True)
        return [RetrievalResult(chunk=chunk, score=score) for score, chunk in sorted_results]

    @staticmethod
    def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
        numerator = sum(x * y for x, y in zip(a, b))
        denom_a = sum(x * x for x in a) ** 0.5 or 1.0
        denom_b = sum(x * x for x in b) ** 0.5 or 1.0
        return numerator / (denom_a * denom_b)


__all__ = ["InMemoryEmbeddingStore"]
