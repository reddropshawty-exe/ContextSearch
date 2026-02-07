"""Сценарий использования для семантического поиска по индексированному содержимому."""
from __future__ import annotations

import logging
from typing import Iterable

from rank_bm25 import BM25Okapi

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

from application.use_cases.embedding_utils import get_embedder_for_spec

logger = logging.getLogger(__name__)


def search(
    query_text: str,
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
    use_bm25: bool = True,
    ranking_mode: str = "vector",
) -> list[RetrievalResult]:
    """Искать документы, релевантные заданному запросу."""

    logger.info("Поиск: запрос='%s', ранжирование=%s, bm25=%s", query_text, ranking_mode, use_bm25)
    query = Query(text=query_text)
    expanded_queries: Iterable[Query] = query_rewriter.rewrite(query) or [query]
    logger.info("Переписанные запросы: %s", [q.text for q in expanded_queries])

    doc_specs = [spec for spec in embedding_specs if spec.level == "document"]
    chunk_specs = [spec for spec in embedding_specs if spec.level == "chunk"]
    candidate_docs: dict[str, float] = {}
    bm25_scores = _bm25_document_scores(query_text, document_repository) if use_bm25 else {}
    if bm25_scores:
        _log_bm25_scores(bm25_scores)

    for spec in doc_specs:
        embedder = get_embedder_for_spec(spec, embedders)
        for rewritten in expanded_queries:
            query_embedding = embedder.embed_query(rewritten)
            ann_results = embedding_store.search(spec, query_embedding, top_k=top_k)
            ann_ids = [ann_id for ann_id, _ in ann_results]
            object_ids = embedding_record_repository.find_object_ids(spec.id, ann_ids)
            logger.info("Документный поиск: spec=%s, запрос='%s', ann_ids=%s", spec.id, rewritten.text, ann_ids)
            for (ann_id, score), doc_id in zip(ann_results, object_ids):
                logger.debug("Сравнение документа: doc_id=%s, ann_id=%s, score=%.4f", doc_id, ann_id, score)
                candidate_docs[doc_id] = max(candidate_docs.get(doc_id, 0.0), score)

    doc_specs = [spec for spec in embedding_specs if spec.level == "document"]
    chunk_specs = [spec for spec in embedding_specs if spec.level == "chunk"]
    candidate_docs: dict[str, float] = {}
    bm25_scores = _bm25_document_scores(query_text, document_repository) if use_bm25 else {}

    for spec in doc_specs:
        embedder = get_embedder_for_spec(spec, embedders)
        for rewritten in expanded_queries:
            query_embedding = embedder.embed_query(rewritten)
            ann_results = embedding_store.search(spec, query_embedding, top_k=top_k)
            ann_ids = [ann_id for ann_id, _ in ann_results]
            object_ids = embedding_record_repository.find_object_ids(spec.id, ann_ids)
            for (ann_id, score), doc_id in zip(ann_results, object_ids):
                candidate_docs[doc_id] = max(candidate_docs.get(doc_id, 0.0), score)

    aggregated_results: list[RetrievalResult] = []
    doc_rrf_scores = _rrf_scores(
        _ranked_ids(candidate_docs),
        _ranked_ids(bm25_scores),
    )
    if doc_rrf_scores:
        logger.debug("RRF оценки документов: %s", doc_rrf_scores)
    for rewritten in expanded_queries:
        for spec in chunk_specs:
            embedder = get_embedder_for_spec(spec, embedders)
            query_embedding = embedder.embed_query(rewritten)
            ann_results = embedding_store.search(spec, query_embedding, top_k=top_k * 2)
            ann_ids = [ann_id for ann_id, _ in ann_results]
            chunk_ids = embedding_record_repository.find_object_ids(spec.id, ann_ids)
            logger.info("Чанковый поиск: spec=%s, запрос='%s', ann_ids=%s", spec.id, rewritten.text, ann_ids)
            for (ann_id, score), chunk_id in zip(ann_results, chunk_ids):
                chunk = chunk_repository.get(chunk_id)
                if chunk is None:
                    continue
                doc_vector_score = candidate_docs.get(chunk.document_id)
                bm25_score = bm25_scores.get(chunk.document_id)
                doc_rrf_score = doc_rrf_scores.get(chunk.document_id, 0.0)
                ranking_score = score
                if ranking_mode == "bm25":
                    ranking_score = float(bm25_score or 0.0)
                logger.debug(
                    "Сравнение чанка: chunk_id=%s, doc_id=%s, ann_id=%s, chunk_score=%.4f, doc_score=%s, bm25=%s",
                    chunk_id,
                    chunk.document_id,
                    ann_id,
                    score,
                    doc_vector_score,
                    bm25_score,
                )
                aggregated_results.append(
                    RetrievalResult(
                        document_id=chunk.document_id,
                        score=ranking_score,
                        chunk=chunk,
                        chunk_text_preview=chunk.text[:200],
                        metadata={
                            "chunk_score": score,
                            "document_vector_score": doc_vector_score,
                            "bm25_score": bm25_score,
                            "document_rrf_score": doc_rrf_score,
                        },
                    )
                )

    deduped = _deduplicate_results(aggregated_results)
    for result in deduped:
        result.document = document_repository.get(result.document_id)
    reranked = reranker.rerank(query, deduped)
    logger.info("Итог: результатов=%d", len(reranked))
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


def _bm25_document_scores(query_text: str, document_repository: DocumentRepository) -> dict[str, float]:
    documents = document_repository.list()
    if not documents:
        return {}
    corpus = [_tokenize(doc.content) for doc in documents]
    if not any(corpus):
        return {}
    bm25 = BM25Okapi(corpus)
    query_tokens = _tokenize(query_text)
    if not query_tokens:
        return {}
    scores = bm25.get_scores(query_tokens)
    return {doc.id: float(score) for doc, score in zip(documents, scores)}


def _log_bm25_scores(scores: dict[str, float]) -> None:
    for doc_id, score in scores.items():
        logger.debug("BM25 сравнение: doc_id=%s, score=%.4f", doc_id, score)


def _tokenize(text: str) -> list[str]:
    return [token for token in text.lower().split() if token]


def _ranked_ids(scores: dict[str, float]) -> list[str]:
    return [doc_id for doc_id, _score in sorted(scores.items(), key=lambda item: item[1], reverse=True)]


def _rrf_scores(*rankings: list[str], k: int = 60) -> dict[str, float]:
    merged: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            merged[doc_id] = merged.get(doc_id, 0.0) + 1.0 / (k + rank)
    return merged
