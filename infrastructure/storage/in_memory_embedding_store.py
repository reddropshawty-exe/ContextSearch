"""Хранилище эмбеддингов в памяти для демо."""
from __future__ import annotations

import heapq
from typing import Sequence

from domain.entities import EmbeddingSpec
from domain.interfaces import EmbeddingRecordRepository, EmbeddingStore


class InMemoryEmbeddingStore(EmbeddingStore):
    """Хранит эмбеддинги в списках Python и ищет перебором."""

    def __init__(self, record_repository: EmbeddingRecordRepository | None = None) -> None:
        self._entries: dict[str, list[tuple[int, str, list[float]]]] = {}
        self._next_id: dict[str, int] = {}
        self._record_repository = record_repository

    def add(
        self,
        spec: EmbeddingSpec,
        object_type: str,
        object_ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        spec_entries = self._entries.setdefault(spec.id, [])
        next_id = self._next_id.setdefault(spec.id, 0)
        for object_id, embedding in zip(object_ids, embeddings):
            spec_entries.append((next_id, object_id, list(embedding)))
            next_id += 1
        self._next_id[spec.id] = next_id
        if self._record_repository:
            self._record_repository.add_records(spec.id, object_type, object_ids, list(range(next_id - len(object_ids), next_id)))

    def search(
        self,
        spec: EmbeddingSpec,
        query_embedding: Sequence[float],
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        scored: list[tuple[float, int]] = []
        for ann_id, _object_id, embedding in self._entries.get(spec.id, []):
            score = self._cosine_similarity(query_embedding, embedding)
            heapq.heappush(scored, (score, ann_id))
            if len(scored) > top_k:
                heapq.heappop(scored)

        sorted_results = sorted(scored, key=lambda item: item[0], reverse=True)
        return [(ann_id, score) for score, ann_id in sorted_results]

    @staticmethod
    def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
        numerator = sum(x * y for x, y in zip(a, b))
        denom_a = sum(x * x for x in a) ** 0.5 or 1.0
        denom_b = sum(x * x for x in b) ** 0.5 or 1.0
        return numerator / (denom_a * denom_b)


__all__ = ["InMemoryEmbeddingStore"]
