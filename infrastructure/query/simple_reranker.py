"""Переранжировщик, сортирующий результаты по оценке."""
from __future__ import annotations

from typing import Iterable

from domain.entities import RetrievalResult
from domain.interfaces import Reranker


class SimpleReranker(Reranker):
    """Сортирует результаты поиска по убыванию оценки."""

    def rerank(self, query, results: Iterable[RetrievalResult]) -> list[RetrievalResult]:  # pragma: no cover - тривиально
        return sorted(results, key=lambda result: result.score, reverse=True)


__all__ = ["SimpleReranker"]
