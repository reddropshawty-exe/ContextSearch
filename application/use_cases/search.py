"""Сценарий использования для семантического поиска по индексированному содержимому."""
from __future__ import annotations

from typing import Iterable

from domain.entities import Chunk, EmbeddingSpec, Query, RetrievalResult
from domain.interfaces import (
    ChunkRepository,
    DocumentRepository,
    Embedder,
    EmbeddingRecordRepository,
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
    chunk_repository: ChunkRepository,
    embedding_record_repository: EmbeddingRecordRepository,
    embedding_specs: list[EmbeddingSpec],
    query_rewriter: QueryRewriter,
    reranker: Reranker,
    top_k: int = 5,
) -> list[RetrievalResult]:
    """Искать документы, релевантные заданному запросу."""

    query = Query(text=query_text)
    expanded_queries: Iterable[Query] = query_rewriter.rewrite(query) or [query]

    doc_specs = [spec for spec in embedding_specs if spec.level == "document"]
    chunk_specs = [spec for spec in embedding_specs if spec.level == "chunk"]
    candidate_docs: dict[str, float] = {}

    for spec in doc_specs:
        for rewritten in expanded_queries:
            query_embedding = embedder.embed_query(rewritten)
            ann_results = embedding_store.search(spec, query_embedding, top_k=top_k)
            ann_ids = [ann_id for ann_id, _ in ann_results]
            object_ids = embedding_record_repository.find_object_ids(spec.id, ann_ids)
            for (ann_id, score), doc_id in zip(ann_results, object_ids):
                candidate_docs[doc_id] = max(candidate_docs.get(doc_id, 0.0), score)

    aggregated_results: list[RetrievalResult] = []
    for rewritten in expanded_queries:
        query_embedding = embedder.embed_query(rewritten)
        for spec in chunk_specs:
            ann_results = embedding_store.search(spec, query_embedding, top_k=top_k * 2)
            ann_ids = [ann_id for ann_id, _ in ann_results]
            chunk_ids = embedding_record_repository.find_object_ids(spec.id, ann_ids)
            for (ann_id, score), chunk_id in zip(ann_results, chunk_ids):
                chunk = chunk_repository.get(chunk_id)
                if chunk is None:
                    continue
                if candidate_docs and chunk.document_id not in candidate_docs:
                    continue
                aggregated_results.append(
                    RetrievalResult(
                        document_id=chunk.document_id,
                        score=score,
                        chunk=chunk,
                        chunk_text_preview=chunk.text[:200],
                    )
                )

    deduped = _deduplicate_results(aggregated_results)
    for result in deduped:
        result.document = document_repository.get(result.document_id)
    reranked = reranker.rerank(query, deduped)
    return reranked[:top_k]


def _deduplicate_results(results: list[RetrievalResult]) -> list[RetrievalResult]:
    merged: dict[str, RetrievalResult] = {}
    for result in results:
        if result.chunk is None:
            continue
        chunk_id = result.chunk.id
        existing = merged.get(chunk_id)
        if existing is None or result.score > existing.score:
            merged[chunk_id] = result
    return list(merged.values())
