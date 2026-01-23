"""Вспомогательный сценарий использования для экспериментов поиска."""
from __future__ import annotations

from statistics import mean
from typing import Iterable, Sequence

from domain.entities import Query, RetrievalResult
from domain.interfaces import (
    DocumentRepository,
    Embedder,
    EmbeddingStore,
    QueryRewriter,
    Reranker,
)


def run_experiment(
    queries: Sequence[str],
    *,
    embedder: Embedder,
    embedding_store: EmbeddingStore,
    document_repository: DocumentRepository,
    query_rewriter: QueryRewriter,
    reranker: Reranker,
    top_k: int = 5,
) -> dict[str, float | list[RetrievalResult]]:
    """Запустить поиск по нескольким запросам и собрать метрики."""

    per_query_results: dict[str, list[RetrievalResult]] = {}
    recall_values: list[float] = []

    for query_text in queries:
        query = Query(text=query_text)
        rewritten_queries: Iterable[Query] = query_rewriter.rewrite(query) or [query]

        aggregated_results: list[RetrievalResult] = []
        for rewritten in rewritten_queries:
            query_embedding = embedder.embed_query(rewritten)
            results = embedding_store.search(query_embedding, top_k=top_k)
            for result in results:
                if result.document is None:
                    result.document = document_repository.get(result.chunk.document_id)
            aggregated_results.extend(results)

        deduped = _deduplicate_results(aggregated_results)
        ranked = reranker.rerank(query, deduped)
        per_query_results[query_text] = ranked[:top_k]
        recall_values.append(1.0 if ranked else 0.0)

    return {
        "per_query_results": per_query_results,
        "average_recall": mean(recall_values) if recall_values else 0.0,
    }


def _deduplicate_results(results: list[RetrievalResult]) -> list[RetrievalResult]:
    merged: dict[str, RetrievalResult] = {}
    for result in results:
        chunk_id = result.chunk.id
        existing = merged.get(chunk_id)
        if existing is None or result.score > existing.score:
            merged[chunk_id] = result
    return list(merged.values())
