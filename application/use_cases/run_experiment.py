"""Вспомогательный сценарий использования для экспериментов поиска."""
from __future__ import annotations

from statistics import mean
from typing import Iterable, Sequence

from domain.entities import EmbeddingSpec, Query, RetrievalResult
from domain.interfaces import (
    ChunkRepository,
    DocumentRepository,
    Embedder,
    EmbeddingRecordRepository,
    EmbeddingStore,
    QueryRewriter,
    Reranker,
)

from application.use_cases.embedding_utils import get_embedder_for_spec


def run_experiment(
    queries: Sequence[str],
    *,
    embedders: dict[str, Embedder],
    embedding_store: EmbeddingStore,
    document_repository: DocumentRepository,
    chunk_repository: ChunkRepository,
    embedding_record_repository: EmbeddingRecordRepository,
    embedding_specs: list[EmbeddingSpec],
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
        doc_specs = [spec for spec in embedding_specs if spec.level == "document"]
        chunk_specs = [spec for spec in embedding_specs if spec.level == "chunk"]
        candidate_docs: dict[str, float] = {}

        for spec in doc_specs:
            embedder = get_embedder_for_spec(spec, embedders)
            for rewritten in rewritten_queries:
                query_embedding = embedder.embed_query(rewritten)
                ann_results = embedding_store.search(spec, query_embedding, top_k=top_k)
                ann_ids = [ann_id for ann_id, _ in ann_results]
                object_ids = embedding_record_repository.find_object_ids(spec.id, ann_ids)
                for (ann_id, score), doc_id in zip(ann_results, object_ids):
                    candidate_docs[doc_id] = max(candidate_docs.get(doc_id, 0.0), score)

        for rewritten in rewritten_queries:
            for spec in chunk_specs:
                embedder = get_embedder_for_spec(spec, embedders)
                query_embedding = embedder.embed_query(rewritten)
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
        if result.chunk is None:
            continue
        chunk_id = result.chunk.id
        existing = merged.get(chunk_id)
        if existing is None or result.score > existing.score:
            merged[chunk_id] = result
    return list(merged.values())
