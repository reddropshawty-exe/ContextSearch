"""Настройка зависимостей для приложения ContextSearch."""
from __future__ import annotations

import os
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
from infrastructure.embedding.sentence_transformers_embedder import (
    SentenceTransformersConfig,
    SentenceTransformersEmbedder,
)
from infrastructure.query.llm_rewriter import LLMQueryRewriter, LLMRewriterConfig
from infrastructure.query.simple_reranker import SimpleReranker
from infrastructure.query.simple_rewriter import SimpleQueryRewriter
from infrastructure.repositories.sqlite_document_repository import SqliteDocumentRepository
from infrastructure.splitting.fixed_window_splitter import FixedWindowSplitter
from infrastructure.storage.faiss_embedding_store import FaissCollection, FaissEmbeddingStore
from infrastructure.storage.in_memory_embedding_store import InMemoryEmbeddingStore
from infrastructure.text_extraction.docx_extractor import DocxExtractor
from infrastructure.text_extraction.html_extractor import HtmlExtractor
from infrastructure.text_extraction.pdf_extractor import PdfExtractor
from infrastructure.text_extraction.plain_text_extractor import PlainTextExtractor


ExtractorName = Literal["pdf", "plain", "html", "docx"]
EmbedderName = Literal[
    "hash-minilm",
    "hash-mean-word",
    "hash-char-ngram",
    "all-minilm",
    "all-mpnet",
    "multilingual-e5-base",
    "embedding-gemma",
]
RewriterName = Literal["simple", "llm"]
EmbeddingStoreName = Literal["in_memory", "faiss"]
ProfileName = Literal["stable", "experimental"]


@dataclass(slots=True)
class Container:
    """Простой контейнер с конкретными инфраструктурными реализациями."""

    extractor: TextExtractor
    splitter: ChunkSplitter
    embedder: Embedder
    embedding_store: EmbeddingStore
    document_repository: DocumentRepository
    query_rewriter: QueryRewriter
    reranker: Reranker


@dataclass(slots=True)
class ContainerConfig:
    """Конфигурация выбора экстракторов и эмбеддеров."""

    extractor: ExtractorName = "pdf"
    embedder: EmbedderName = "all-minilm"
    rewriter: RewriterName = "simple"
    embedding_store: EmbeddingStoreName = "faiss"
    profile: ProfileName = "stable"
    collection_id: str | None = None
    device: str = "cpu"
    normalize_embeddings: bool = True
    embeddinggemma_model: str = "google/embeddinggemma-300m"
    local_rewriter_model: str = "google/flan-t5-small"
    safe_mode: bool = False


_EXTRACTOR_FACTORIES: dict[ExtractorName, Callable[[], TextExtractor]] = {
    "pdf": PdfExtractor,
    "plain": PlainTextExtractor,
    "html": HtmlExtractor,
    "docx": DocxExtractor,
}

_EMBEDDER_FACTORIES: dict[EmbedderName, Callable[[], Embedder]] = {
    "hash-minilm": MiniLMEmbedder,
    "hash-mean-word": MeanWordHashEmbedder,
    "hash-char-ngram": CharacterNgramEmbedder,
}


def build_default_container(config: ContainerConfig | None = None) -> Container:
    """Создать стандартный набор инфраструктурных компонентов."""

    cfg = config or ContainerConfig()
    if _safe_mode_enabled(cfg):
        cfg = _apply_safe_mode(cfg)
    try:
        extractor = _EXTRACTOR_FACTORIES[cfg.extractor]()
    except KeyError as exc:  # pragma: no cover - защитный код
        raise ValueError(f"Неизвестный экстрактор '{cfg.extractor}'") from exc
    splitter = FixedWindowSplitter()
    embedder = _build_embedder(cfg)
    embedding_store = _build_embedding_store(cfg, embedder)
    document_repository = SqliteDocumentRepository(db_path="contextsearch.db")
    query_rewriter = _build_rewriter(cfg)
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


def _build_embedder(config: ContainerConfig) -> Embedder:
    if config.embedder in _EMBEDDER_FACTORIES:
        return _EMBEDDER_FACTORIES[config.embedder]()

    model_config = _sentence_model_config(config)
    return SentenceTransformersEmbedder(model_config)


def _sentence_model_config(config: ContainerConfig) -> SentenceTransformersConfig:
    query_prefix = None
    passage_prefix = None
    model_name = config.embeddinggemma_model
    if config.embedder == "all-minilm":
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
    elif config.embedder == "all-mpnet":
        model_name = "sentence-transformers/all-mpnet-base-v2"
    elif config.embedder == "multilingual-e5-base":
        model_name = "intfloat/multilingual-e5-base"
        query_prefix = "query: "
        passage_prefix = "passage: "
    elif config.embedder == "embedding-gemma":
        model_name = config.embeddinggemma_model
    return SentenceTransformersConfig(
        model_name=model_name,
        device=config.device,
        normalize_embeddings=config.normalize_embeddings,
        query_prefix=query_prefix,
        passage_prefix=passage_prefix,
    )


def _build_embedding_store(config: ContainerConfig, embedder: Embedder) -> EmbeddingStore:
    if config.embedding_store == "in_memory":
        return InMemoryEmbeddingStore(collection_id="in-memory")

    collection_id = config.collection_id or ("stable" if config.profile == "stable" else "experimental")
    collection = FaissCollection(
        collection_id=collection_id,
        model_id=embedder.model_id,
        dimension=embedder.dimension,
    )
    return FaissEmbeddingStore(
        collection=collection,
        index_root="indexes",
        db_path="contextsearch.db",
        normalize_embeddings=config.normalize_embeddings,
    )


def _build_rewriter(config: ContainerConfig) -> QueryRewriter:
    if config.rewriter == "llm":
        return LLMQueryRewriter(
            LLMRewriterConfig(
                model=config.local_rewriter_model,
            )
        )
    return SimpleQueryRewriter()


def _safe_mode_enabled(config: ContainerConfig) -> bool:
    if config.safe_mode:
        return True
    return os.getenv("CONTEXTSEARCH_SAFE_MODE") == "1"


def _apply_safe_mode(config: ContainerConfig) -> ContainerConfig:
    return ContainerConfig(
        extractor=config.extractor,
        embedder="hash-minilm",
        rewriter="simple",
        embedding_store="in_memory",
        profile=config.profile,
        collection_id=config.collection_id,
        device=config.device,
        normalize_embeddings=config.normalize_embeddings,
        embeddinggemma_model=config.embeddinggemma_model,
        local_rewriter_model=config.local_rewriter_model,
        safe_mode=True,
    )


__all__ = ["Container", "ContainerConfig", "build_default_container"]
