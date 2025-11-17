"""Reranker that sorts results by score."""
from __future__ import annotations

from typing import Iterable

from domain.entities import RetrievalResult
from domain.interfaces import Reranker


class SimpleReranker(Reranker):
    """Sort retrieval results by score descending."""

    def rerank(self, query, results: Iterable[RetrievalResult]) -> list[RetrievalResult]:  # pragma: no cover - trivial
        return sorted(results, key=lambda result: result.score, reverse=True)


__all__ = ["SimpleReranker"]
