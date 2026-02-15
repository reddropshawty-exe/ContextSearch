"""Слой FastAPI, который предоставляет операции индексации и поиска."""
from __future__ import annotations

from fastapi import FastAPI, Query as FastAPIQuery
from pydantic import BaseModel

from application.use_cases.ingest_documents import ingest_documents
from application.use_cases.search import search
from application.evaluation.service import (
    create_experiment_config,
    create_test_case,
    create_test_suite,
    run_and_save_experiment,
)
from domain.entities import Document
from infrastructure.config import build_default_container
from ui.logging_utils import setup_logging

setup_logging()

app = FastAPI(title="API ContextSearch")
container = build_default_container()


class DocumentPayload(BaseModel):
    id: str | None = None
    content: str


class IngestRequest(BaseModel):
    documents: list[DocumentPayload]


class IngestResponse(BaseModel):
    ingested: int


class SearchResponse(BaseModel):
    query: str
    results: list[dict]


class EvalCasePayload(BaseModel):
    query_text: str
    relevant_document_ids: list[str]
    source: str = "user"


class EvalSuitePayload(BaseModel):
    name: str
    description: str = ""
    cases: list[EvalCasePayload]


class EvalConfigPayload(BaseModel):
    embedding_spec_id: str
    store_type: str
    use_bm25: bool = True
    ranking_mode: str = "hybrid_rrf"
    query_rewriter: str = "none"
    top_k: int = 10
    rrf_k: int = 60


class EvalRunPayload(BaseModel):
    suite_id: str
    config_id: str


@app.post("/ingest", response_model=IngestResponse)
def ingest_endpoint(payload: IngestRequest) -> IngestResponse:
    documents = [(doc.id, doc.content) for doc in payload.documents]
    ingest_documents(
        documents,
        extractor=container.extractor,
        splitter=container.splitter,
        embedders=container.embedders,
        embedding_store=container.embedding_store,
        document_repository=container.document_repository,
        chunk_repository=container.chunk_repository,
        embedding_specs=container.embedding_specs,
        bm25_index=container.bm25_index,
    )
    return IngestResponse(ingested=len(documents))


@app.get("/documents", response_model=list[DocumentPayload])
def documents_endpoint() -> list[DocumentPayload]:
    documents: list[Document] = container.document_repository.list()
    return [DocumentPayload(id=doc.id, content=doc.content) for doc in documents]


@app.get("/search", response_model=SearchResponse)
def search_endpoint(q: str = FastAPIQuery(..., description="Запрос пользователя")) -> SearchResponse:
    results = search(
        q,
        embedders=container.embedders,
        embedding_store=container.embedding_store,
        document_repository=container.document_repository,
        chunk_repository=container.chunk_repository,
        embedding_record_repository=container.embedding_record_repository,
        embedding_specs=container.embedding_specs,
        bm25_index=container.bm25_index,
        query_rewriter=container.query_rewriter,
        reranker=container.reranker,
    )
    serialized = [
        {
            "chunk_id": result.chunk.id if result.chunk else None,
            "document_id": result.document_id,
            "chunk_score": result.metadata.get("chunk_score", result.score),
            "document_score": result.metadata.get("document_vector_score"),
            "bm25_score": result.metadata.get("bm25_score"),
            "text": result.chunk.text if result.chunk else None,
        }
        for result in results
    ]
    return SearchResponse(query=q, results=serialized)


@app.post("/evaluation/suites")
def create_eval_suite(payload: EvalSuitePayload) -> dict:
    if container.evaluation_repository is None:
        return {"error": "evaluation repository unavailable"}
    suite = create_test_suite(payload.name, payload.description)
    for case in payload.cases:
        suite.test_cases.append(
            create_test_case(
                case.query_text,
                case.relevant_document_ids,
                source=case.source,
            )
        )
    container.evaluation_repository.upsert_suite(suite)
    return {"suite_id": suite.id, "cases": len(suite.test_cases)}


@app.get("/evaluation/suites")
def list_eval_suites() -> list[dict]:
    if container.evaluation_repository is None:
        return []
    suites = container.evaluation_repository.list_suites()
    return [{"id": s.id, "name": s.name, "description": s.description} for s in suites]


@app.post("/evaluation/configs")
def create_eval_config(payload: EvalConfigPayload) -> dict:
    if container.evaluation_repository is None:
        return {"error": "evaluation repository unavailable"}
    config = create_experiment_config(
        embedding_spec_id=payload.embedding_spec_id,
        store_type=payload.store_type,
        use_bm25=payload.use_bm25,
        ranking_mode=payload.ranking_mode,
        query_rewriter=payload.query_rewriter,
        top_k=payload.top_k,
        rrf_k=payload.rrf_k,
    )
    container.evaluation_repository.upsert_config(config)
    return {"config_id": config.id}


@app.get("/evaluation/configs")
def list_eval_configs() -> list[dict]:
    if container.evaluation_repository is None:
        return []
    cfgs = container.evaluation_repository.list_configs()
    return [
        {
            "id": c.id,
            "embedding_spec_id": c.embedding_spec_id,
            "store_type": c.store_type,
            "use_bm25": c.use_bm25,
            "ranking_mode": c.ranking_mode,
            "query_rewriter": c.query_rewriter,
            "top_k": c.top_k,
            "rrf_k": c.rrf_k,
        }
        for c in cfgs
    ]


@app.post("/evaluation/runs")
def run_evaluation(payload: EvalRunPayload) -> dict:
    if container.evaluation_repository is None:
        return {"error": "evaluation repository unavailable"}
    suite = container.evaluation_repository.get_suite(payload.suite_id)
    if suite is None:
        return {"error": "suite not found"}
    configs = {c.id: c for c in container.evaluation_repository.list_configs()}
    config = configs.get(payload.config_id)
    if config is None:
        return {"error": "config not found"}
    run = run_and_save_experiment(
        suite=suite,
        config=config,
        container=container,
        repository=container.evaluation_repository,
    )
    return {"run_id": run.id, "aggregate_metrics": run.aggregate_metrics}


@app.get("/evaluation/runs")
def list_eval_runs() -> list[dict]:
    if container.evaluation_repository is None:
        return []
    runs = container.evaluation_repository.list_runs()
    return [
        {
            "id": r.id,
            "suite_id": r.test_suite_id,
            "config_id": r.experiment_config_id,
            "created_at": r.created_at.isoformat(),
            "aggregate_metrics": r.aggregate_metrics,
        }
        for r in runs
    ]
