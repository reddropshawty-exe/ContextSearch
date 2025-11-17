"""Dependency wiring for the ContextSearch application."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from domain.interfaces import (
    ChunkSplitter,
    DocumentRepository,
    Embedder,
    EmbeddingStore,
    QueryRewriter,
    Reranker,
    TextExtractor,
)
from infrastructure.embedding.character_ngram_embedder import CharacterNgramEmbedder
from infrastructure.embedding.mean_word_hash_embedder import MeanWordHashEmbedder
from infrastructure.embedding.minilm_embedder import MiniLMEmbedder
from infrastructure.query.simple_reranker import SimpleReranker
from infrastructure.query.simple_rewriter import SimpleQueryRewriter
from infrastructure.repositories.sqlite_document_repository import SqliteDocumentRepository
from infrastructure.splitting.fixed_window_splitter import FixedWindowSplitter
from infrastructure.storage.in_memory_embedding_store import InMemoryEmbeddingStore
from infrastructure.text_extraction.html_extractor import HtmlExtractor
from infrastructure.text_extraction.pdf_extractor import PdfExtractor
from infrastructure.text_extraction.plain_text_extractor import PlainTextExtractor


ExtractorName = Literal["pdf", "plain", "html"]
EmbedderName = Literal["minilm", "mean_word", "char_ngram"]


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


@dataclass(slots=True)
class ContainerConfig:
    """Configuration for selecting extractor/embedder models."""

    extractor: ExtractorName = "pdf"
    embedder: EmbedderName = "minilm"


_EXTRACTOR_FACTORIES: dict[ExtractorName, Callable[[], TextExtractor]] = {
    "pdf": PdfExtractor,
    "plain": PlainTextExtractor,
    "html": HtmlExtractor,
}

_EMBEDDER_FACTORIES: dict[EmbedderName, Callable[[], Embedder]] = {
    "minilm": MiniLMEmbedder,
    "mean_word": MeanWordHashEmbedder,
    "char_ngram": CharacterNgramEmbedder,
}


def build_default_container(config: ContainerConfig | None = None) -> Container:
    """Instantiate the default infrastructure stack."""

    cfg = config or ContainerConfig()
    try:
        extractor = _EXTRACTOR_FACTORIES[cfg.extractor]()
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unknown extractor '{cfg.extractor}'") from exc
    splitter = FixedWindowSplitter()
    try:
        embedder = _EMBEDDER_FACTORIES[cfg.embedder]()
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unknown embedder '{cfg.embedder}'") from exc
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


__all__ = ["Container", "ContainerConfig", "build_default_container"]
