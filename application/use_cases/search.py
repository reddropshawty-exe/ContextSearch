"""Use case that performs semantic search over ingested content."""
from __future__ import annotations

from typing import Iterable

from domain.entities import Query, RetrievalResult
from domain.interfaces import (
    DocumentRepository,
    Embedder,
    EmbeddingStore,
    QueryRewriter,
    Reranker,
)


def search(
    query_text: str,
    *,
    embedder: Embedder,
    embedding_store: EmbeddingStore,
    document_repository: DocumentRepository,
    query_rewriter: QueryRewriter,
    reranker: Reranker,
    top_k: int = 5,
) -> list[RetrievalResult]:
    """Search for documents relevant to the provided query text."""

    query = Query(text=query_text)
    expanded_queries: Iterable[Query] = query_rewriter.rewrite(query) or [query]

    aggregated_results: list[RetrievalResult] = []
    for rewritten in expanded_queries:
        query_embedding = embedder.embed_query(rewritten)
        results = embedding_store.search(query_embedding, top_k=top_k)
        for result in results:
            if result.document is None:
                result.document = document_repository.get(result.chunk.document_id)
        aggregated_results.extend(results)

    deduped = _deduplicate_results(aggregated_results)
    reranked = reranker.rerank(query, deduped)
    return reranked[:top_k]


def _deduplicate_results(results: list[RetrievalResult]) -> list[RetrievalResult]:
    merged: dict[str, RetrievalResult] = {}
    for result in results:
        chunk_id = result.chunk.id
        existing = merged.get(chunk_id)
        if existing is None or result.score > existing.score:
            merged[chunk_id] = result
    return list(merged.values())
