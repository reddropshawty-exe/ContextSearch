from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from application.evaluation.models import (
    CaseRunResult,
    ExperimentConfig,
    ExperimentRun,
    RelevanceLabel,
    TestCase,
    TestSuite,
)
from application.evaluation.runner import run_experiment_suite
from application.use_cases.search import search
from infrastructure.config import Container
from infrastructure.repositories.sqlite_evaluation_repository import SqliteEvaluationRepository


def create_test_suite(name: str, description: str = "") -> TestSuite:
    return TestSuite(id=str(uuid4()), name=name, description=description)


def create_test_case(
    query_text: str,
    relevant_document_ids: list[str],
    *,
    source: str = "user",
    metadata: dict[str, str] | None = None,
    graded_relevance: dict[str, int] | None = None,
) -> TestCase:
    labels = [
        RelevanceLabel(document_id=doc_id, grade=(graded_relevance or {}).get(doc_id, 1))
        for doc_id in relevant_document_ids
    ]
    return TestCase(
        id=str(uuid4()),
        query_text=query_text,
        source=source,
        relevance_labels=labels,
        metadata=metadata or {},
    )


def create_experiment_config(
    *,
    embedding_spec_id: str,
    store_type: str,
    use_bm25: bool,
    ranking_mode: str,
    query_rewriter: str,
    top_k: int = 10,
    rrf_k: int = 60,
    metadata: dict[str, str] | None = None,
) -> ExperimentConfig:
    return ExperimentConfig(
        id=str(uuid4()),
        embedding_spec_id=embedding_spec_id,
        store_type=store_type,
        use_bm25=use_bm25,
        ranking_mode=ranking_mode,
        query_rewriter=query_rewriter,
        top_k=top_k,
        rrf_k=rrf_k,
        metadata=metadata or {},
    )


def run_and_save_experiment(
    *,
    suite: TestSuite,
    config: ExperimentConfig,
    container: Container,
    repository: SqliteEvaluationRepository,
) -> ExperimentRun:
    ranking_mode = "rrf" if config.ranking_mode == "hybrid_rrf" else config.ranking_mode

    def _search_fn(query_text: str, top_k: int) -> list[dict[str, object]]:
        results = search(
            query_text,
            embedders=container.embedders,
            embedding_store=container.embedding_store,
            document_repository=container.document_repository,
            chunk_repository=container.chunk_repository,
            embedding_record_repository=container.embedding_record_repository,
            embedding_specs=container.embedding_specs,
            query_rewriter=container.query_rewriter,
            reranker=container.reranker,
            bm25_index=container.bm25_index,
            top_k=top_k,
            use_bm25=config.use_bm25,
            ranking_mode=ranking_mode,
        )
        serialized: list[dict[str, object]] = []
        for item in results:
            serialized.append(
                {
                    "document_id": item.document_id,
                    "chunk_id": item.chunk.id if item.chunk else None,
                    "score": item.score,
                    "vector_score": item.metadata.get("document_vector_score"),
                    "bm25_score": item.metadata.get("bm25_score"),
                    "chunk_score": item.metadata.get("chunk_score"),
                }
            )
        return serialized

    run = run_experiment_suite(suite, config, _search_fn)
    run.id = str(uuid4())
    run.created_at = datetime.utcnow()
    repository.upsert_suite(suite)
    repository.upsert_config(config)
    repository.save_run(run)
    return run


__all__ = [
    "create_test_suite",
    "create_test_case",
    "create_experiment_config",
    "run_and_save_experiment",
    "CaseRunResult",
    "ExperimentConfig",
    "ExperimentRun",
    "RelevanceLabel",
    "TestCase",
    "TestSuite",
]

