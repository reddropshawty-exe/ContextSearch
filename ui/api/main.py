"""FastAPI layer that exposes ingest/search operations."""
from __future__ import annotations

from fastapi import FastAPI, Query as FastAPIQuery
from pydantic import BaseModel

from application.use_cases.ingest_documents import ingest_documents
from application.use_cases.search import search
from domain.entities import Document
from infrastructure.config import build_default_container

app = FastAPI(title="ContextSearch API")
container = build_default_container()


class DocumentPayload(BaseModel):
    id: str
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
        embedder=container.embedder,
        embedding_store=container.embedding_store,
        document_repository=container.document_repository,
    )
    return IngestResponse(ingested=len(documents))


@app.get("/documents", response_model=list[DocumentPayload])
def documents_endpoint() -> list[DocumentPayload]:
    documents: list[Document] = container.document_repository.list()
    return [DocumentPayload(id=doc.id, content=doc.content) for doc in documents]


@app.get("/search", response_model=SearchResponse)
def search_endpoint(q: str = FastAPIQuery(..., description="User query")) -> SearchResponse:
    results = search(
        q,
        embedder=container.embedder,
        embedding_store=container.embedding_store,
        document_repository=container.document_repository,
        query_rewriter=container.query_rewriter,
        reranker=container.reranker,
    )
    serialized = [
        {
            "chunk_id": result.chunk.id,
            "document_id": result.chunk.document_id,
            "score": result.score,
            "text": result.chunk.text,
        }
        for result in results
    ]
    return SearchResponse(query=q, results=serialized)
