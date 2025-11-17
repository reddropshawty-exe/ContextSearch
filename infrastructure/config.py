"""Dependency wiring for the ContextSearch application."""
from __future__ import annotations

from dataclasses import dataclass

from domain.interfaces import (
    ChunkSplitter,
    DocumentRepository,
    Embedder,
    EmbeddingStore,
    QueryRewriter,
    Reranker,
    TextExtractor,
)
from infrastructure.embedding.minilm_embedder import MiniLMEmbedder
from infrastructure.query.simple_reranker import SimpleReranker
from infrastructure.query.simple_rewriter import SimpleQueryRewriter
from infrastructure.repositories.sqlite_document_repository import SqliteDocumentRepository
from infrastructure.splitting.fixed_window_splitter import FixedWindowSplitter
from infrastructure.storage.in_memory_embedding_store import InMemoryEmbeddingStore
from infrastructure.text_extraction.pdf_extractor import PdfExtractor


@dataclass(slots=True)
class Container:
    """Simple container bundling concrete infrastructure implementations."""

    extractor: TextExtractor
    splitter: ChunkSplitter
    embedder: Embedder
    embedding_store: EmbeddingStore
    document_repository: DocumentRepository
    query_rewriter: QueryRewriter
    reranker: Reranker


def build_default_container() -> Container:
    """Instantiate the default infrastructure stack."""

    extractor = PdfExtractor()
    splitter = FixedWindowSplitter()
    embedder = MiniLMEmbedder()
    embedding_store = InMemoryEmbeddingStore()
    document_repository = SqliteDocumentRepository()
    query_rewriter = SimpleQueryRewriter()
    reranker = SimpleReranker()

    return Container(
        extractor=extractor,
        splitter=splitter,
        embedder=embedder,
        embedding_store=embedding_store,
        document_repository=document_repository,
        query_rewriter=query_rewriter,
        reranker=reranker,
    )


__all__ = ["Container", "build_default_container"]
