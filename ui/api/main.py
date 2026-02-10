"""Слой FastAPI, который предоставляет операции индексации и поиска."""
from __future__ import annotations

from fastapi import FastAPI, Query as FastAPIQuery
from pydantic import BaseModel

from application.use_cases.ingest_documents import ingest_documents
from application.use_cases.search import search
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
