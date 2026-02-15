"""Сценарий использования для семантического поиска по индексированному содержимому."""
from __future__ import annotations

import logging
from dataclasses import dataclass
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

from application.services.bm25_index import BM25Index
from application.use_cases.embedding_utils import get_embedder_for_spec
from infrastructure.storage.in_memory_embedding_store import InMemoryEmbeddingStore

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _ChunkHit:
    chunk: Chunk
    score: float


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
    bm25_index: BM25Index | None = None,
    top_k: int = 5,
    use_bm25: bool = True,
    ranking_mode: str = "rrf",
) -> list[RetrievalResult]:
    """Искать документы, релевантные заданному запросу."""

    logger.info("Поиск: запрос='%s', ранжирование=%s, bm25=%s", query_text, ranking_mode, use_bm25)
    query = Query(text=query_text)
    expanded_queries: Iterable[Query] = query_rewriter.rewrite(query) or [query]
    expanded_queries = list(expanded_queries)
    logger.info("Переписанные запросы: %s", [q.text for q in expanded_queries])

    doc_specs = [spec for spec in embedding_specs if spec.level == "document"]
    chunk_specs = [spec for spec in embedding_specs if spec.level == "chunk"]

    _ensure_in_memory_index(
        embedding_store,
        embedders,
        embedding_record_repository,
        document_repository,
        chunk_repository,
        doc_specs,
        chunk_specs,
    )

    vector_doc_scores: dict[str, float] = {}
    best_chunks: dict[str, _ChunkHit] = {}

    for spec in doc_specs:
        embedder = get_embedder_for_spec(spec, embedders)
        for rewritten in expanded_queries:
            query_embedding = embedder.embed_query(rewritten)
            ann_results = embedding_store.search(spec, query_embedding, top_k=max(top_k * 3, 20))
            ann_ids = [ann_id for ann_id, _ in ann_results]
            object_ids = embedding_record_repository.find_object_ids(spec.id, ann_ids)
            logger.info("Документный поиск: spec=%s, запрос='%s', ann_ids=%s", spec.id, rewritten.text, ann_ids)
            for (ann_id, score), doc_id in zip(ann_results, object_ids):
                logger.debug("Сравнение документа: doc_id=%s, ann_id=%s, score=%.4f", doc_id, ann_id, score)
                vector_doc_scores[doc_id] = max(vector_doc_scores.get(doc_id, 0.0), score)

    for spec in chunk_specs:
        embedder = get_embedder_for_spec(spec, embedders)
        for rewritten in expanded_queries:
            query_embedding = embedder.embed_query(rewritten)
            ann_results = embedding_store.search(spec, query_embedding, top_k=max(top_k * 8, 50))
            ann_ids = [ann_id for ann_id, _ in ann_results]
            chunk_ids = embedding_record_repository.find_object_ids(spec.id, ann_ids)
            logger.info("Чанковый поиск: spec=%s, запрос='%s', ann_ids=%s", spec.id, rewritten.text, ann_ids)
            for (ann_id, score), chunk_id in zip(ann_results, chunk_ids):
                chunk = chunk_repository.get(chunk_id)
                if chunk is None:
                    continue
                logger.debug(
                    "Сравнение чанка: chunk_id=%s, doc_id=%s, ann_id=%s, chunk_score=%.4f",
                    chunk_id,
                    chunk.document_id,
                    ann_id,
                    score,
                )
                prev = best_chunks.get(chunk.document_id)
                if prev is None or score > prev.score:
                    best_chunks[chunk.document_id] = _ChunkHit(chunk=chunk, score=score)
                vector_doc_scores[chunk.document_id] = max(vector_doc_scores.get(chunk.document_id, 0.0), score)

    bm25_scores: dict[str, float] = {}
    if use_bm25:
        docs = document_repository.list()
        bm25_index = bm25_index or BM25Index()
        bm25_index.update_documents(docs)
        bm25_scores = bm25_index.scores(query_text)
        for doc_id, score in bm25_scores.items():
            logger.debug("BM25 сравнение: doc_id=%s, score=%.4f", doc_id, score)

    doc_score = _merge_scores(vector_doc_scores, bm25_scores, ranking_mode=ranking_mode)
    doc_ids_ranked = [doc_id for doc_id, _ in sorted(doc_score.items(), key=lambda item: item[1], reverse=True)]

    results: list[RetrievalResult] = []
    for doc_id in doc_ids_ranked:
        document = document_repository.get(doc_id)
        hit = best_chunks.get(doc_id)
        chunk = hit.chunk if hit else None
        chunk_score = hit.score if hit else None
        result = RetrievalResult(
            document_id=doc_id,
            score=float(doc_score[doc_id]),
            chunk=chunk,
            chunk_text_preview=(chunk.text[:200] if chunk else None),
            document=document,
            metadata={
                "chunk_score": chunk_score,
                "document_vector_score": vector_doc_scores.get(doc_id),
                "bm25_score": bm25_scores.get(doc_id),
                "ranking_mode": ranking_mode,
            },
        )
        results.append(result)

    reranked = reranker.rerank(query, results)
    logger.info("Итог: документов=%d", len(reranked))
    return reranked[:top_k]


def _merge_scores(vector_scores: dict[str, float], bm25_scores: dict[str, float], *, ranking_mode: str) -> dict[str, float]:
    if ranking_mode == "vector":
        return dict(vector_scores)
    if ranking_mode == "bm25":
        return dict(bm25_scores)

    vector_ranking = _ranked_ids(vector_scores)
    bm25_ranking = _ranked_ids(bm25_scores)
    rrf = _rrf_scores(vector_ranking, bm25_ranking)
    # fallback: even when one ranking misses doc, keep it discoverable
    for doc_id, score in vector_scores.items():
        rrf.setdefault(doc_id, score * 1e-6)
    for doc_id, score in bm25_scores.items():
        rrf.setdefault(doc_id, score * 1e-6)
    return rrf


def _ensure_in_memory_index(
    embedding_store: EmbeddingStore,
    embedders: dict[str, Embedder],
    embedding_record_repository: EmbeddingRecordRepository,
    document_repository: DocumentRepository,
    chunk_repository: ChunkRepository,
    doc_specs: list[EmbeddingSpec],
    chunk_specs: list[EmbeddingSpec],
) -> None:
    if not isinstance(embedding_store, InMemoryEmbeddingStore):
        return

    for spec in doc_specs:
        if embedding_store.has_entries(spec.id):
            continue
        doc_ids = embedding_record_repository.list_object_ids(spec.id, "document")
        documents = [document_repository.get(doc_id) for doc_id in doc_ids] if doc_ids else document_repository.list()
        texts = [doc.content for doc in documents if doc and doc.content]
        if not texts:
            continue
        embedder = get_embedder_for_spec(spec, embedders)
        logger.info("Восстановление in-memory индекса документов: spec=%s, count=%d", spec.id, len(texts))
        embeddings = embedder.embed_texts(texts)
        valid_ids = [doc.id for doc in documents if doc and doc.content]
        embedding_store.add(spec, "document", valid_ids, embeddings)

    for spec in chunk_specs:
        if embedding_store.has_entries(spec.id):
            continue
        chunk_ids = embedding_record_repository.list_object_ids(spec.id, "chunk")
        chunks = [chunk_repository.get(chunk_id) for chunk_id in chunk_ids] if chunk_ids else chunk_repository.list()
        texts = [chunk.text for chunk in chunks if chunk and chunk.text]
        if not texts:
            continue
        embedder = get_embedder_for_spec(spec, embedders)
        logger.info("Восстановление in-memory индекса чанков: spec=%s, count=%d", spec.id, len(texts))
        embeddings = embedder.embed_texts(texts)
        valid_ids = [chunk.id for chunk in chunks if chunk and chunk.text]
        embedding_store.add(spec, "chunk", valid_ids, embeddings)


def _ranked_ids(scores: dict[str, float]) -> list[str]:
    return [doc_id for doc_id, _score in sorted(scores.items(), key=lambda item: item[1], reverse=True)]


def _rrf_scores(*rankings: list[str], k: int = 60) -> dict[str, float]:
    merged: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            merged[doc_id] = merged.get(doc_id, 0.0) + 1.0 / (k + rank)
    return merged
