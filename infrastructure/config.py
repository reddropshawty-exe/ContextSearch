"""Настройка зависимостей для приложения ContextSearch."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Literal

from domain.entities import EmbeddingSpec
from domain.interfaces import (
    ChunkRepository,
    ChunkSplitter,
    DocumentRepository,
    Embedder,
    EmbeddingRecordRepository,
    EmbeddingSpecRepository,
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
from infrastructure.repositories.sqlite_chunk_repository import SqliteChunkRepository
from infrastructure.repositories.sqlite_document_repository import SqliteDocumentRepository
from infrastructure.repositories.sqlite_embedding_record_repository import SqliteEmbeddingRecordRepository
from infrastructure.repositories.sqlite_embedding_spec_repository import SqliteEmbeddingSpecRepository
from infrastructure.splitting.fixed_window_splitter import FixedWindowSplitter
from infrastructure.storage.hnsw_embedding_store import HnswEmbeddingStore
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
EmbeddingStoreName = Literal["in_memory", "hnsw"]
ProfileName = Literal["stable", "experimental"]


@dataclass(slots=True)
class Container:
    """Простой контейнер с конкретными инфраструктурными реализациями."""

    extractor: TextExtractor
    splitter: ChunkSplitter
    embedder: Embedder
    embedders: dict[EmbedderName, Embedder]
    embedding_store: EmbeddingStore
    document_repository: DocumentRepository
    chunk_repository: ChunkRepository
    embedding_spec_repository: EmbeddingSpecRepository
    embedding_record_repository: EmbeddingRecordRepository
    query_rewriter: QueryRewriter
    reranker: Reranker
    embedding_specs: list[EmbeddingSpec]


@dataclass(slots=True)
class ContainerConfig:
    """Конфигурация выбора экстракторов и эмбеддеров."""

    extractor: ExtractorName = "pdf"
    embedder: EmbedderName = "all-minilm"
    rewriter: RewriterName = "simple"
    embedding_store: EmbeddingStoreName = "hnsw"
    profile: ProfileName = "stable"
    collection_id: str | None = None
    device: str = "cpu"
    normalize_embeddings: bool = True
    embeddinggemma_model: str = "google/embeddinggemma-300m"
    local_rewriter_model: str = "google/flan-t5-small"
    safe_mode: bool = False
    embedder_models: tuple[EmbedderName, ...] = ("all-minilm",)


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
    cfg.embedder_models = _ensure_primary_embedder(cfg.embedder_models, cfg.embedder)
    try:
        extractor = _EXTRACTOR_FACTORIES[cfg.extractor]()
    except KeyError as exc:  # pragma: no cover - защитный код
        raise ValueError(f"Неизвестный экстрактор '{cfg.extractor}'") from exc
    splitter = FixedWindowSplitter()
    embedders = _build_embedders(cfg)
    embedder = embedders[cfg.embedder]
    document_repository = SqliteDocumentRepository(db_path="contextsearch.db")
    chunk_repository = SqliteChunkRepository(db_path="contextsearch.db")
    embedding_spec_repository = SqliteEmbeddingSpecRepository(db_path="contextsearch.db")
    embedding_record_repository = SqliteEmbeddingRecordRepository(db_path="contextsearch.db")
    embedding_specs = _build_embedding_specs(cfg, embedders)
    for spec in embedding_specs:
        embedding_spec_repository.upsert(spec)
    embedding_store = _build_embedding_store(cfg, embedding_record_repository)
    query_rewriter = _build_rewriter(cfg)
    reranker = SimpleReranker()

    return Container(
        extractor=extractor,
        splitter=splitter,
        embedder=embedder,
        embedders=embedders,
        embedding_store=embedding_store,
        document_repository=document_repository,
        chunk_repository=chunk_repository,
        embedding_spec_repository=embedding_spec_repository,
        embedding_record_repository=embedding_record_repository,
        query_rewriter=query_rewriter,
        reranker=reranker,
        embedding_specs=embedding_specs,
    )


def _build_embedder(config: ContainerConfig) -> Embedder:
    return _build_embedder_for_model(config, config.embedder)


def _build_embedders(config: ContainerConfig) -> dict[EmbedderName, Embedder]:
    embedders: dict[EmbedderName, Embedder] = {}
    for model_name in config.embedder_models:
        embedders[model_name] = _build_embedder_for_model(config, model_name)
    return embedders


def _build_embedder_for_model(config: ContainerConfig, model_name: EmbedderName) -> Embedder:
    if model_name in _EMBEDDER_FACTORIES:
        return _EMBEDDER_FACTORIES[model_name]()
    model_config = _sentence_model_config_for(model_name, config)
    return SentenceTransformersEmbedder(model_config)


def _sentence_model_config_for(model_name: EmbedderName, config: ContainerConfig) -> SentenceTransformersConfig:
    query_prefix = None
    passage_prefix = None
    resolved_name = config.embeddinggemma_model
    if model_name == "all-minilm":
        resolved_name = "sentence-transformers/all-MiniLM-L6-v2"
    elif model_name == "all-mpnet":
        resolved_name = "sentence-transformers/all-mpnet-base-v2"
    elif model_name == "multilingual-e5-base":
        resolved_name = "intfloat/multilingual-e5-base"
        query_prefix = "query: "
        passage_prefix = "passage: "
    elif model_name == "embedding-gemma":
        resolved_name = config.embeddinggemma_model
    return SentenceTransformersConfig(
        model_name=resolved_name,
        device=config.device,
        normalize_embeddings=config.normalize_embeddings,
        query_prefix=query_prefix,
        passage_prefix=passage_prefix,
    )


def _build_embedding_store(
    config: ContainerConfig,
    record_repository: EmbeddingRecordRepository,
) -> EmbeddingStore:
    if config.embedding_store == "in_memory":
        return InMemoryEmbeddingStore(record_repository=record_repository)
    return HnswEmbeddingStore(index_root="indexes", record_repository=record_repository)


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
        embedder_models=("hash-minilm",),
    )


def _build_embedding_specs(
    config: ContainerConfig,
    embedders: dict[EmbedderName, Embedder],
) -> list[EmbeddingSpec]:
    specs: list[EmbeddingSpec] = []
    metric = "cosine"
    for model_name in config.embedder_models:
        embedder = embedders[model_name]
        dimension = embedder.dimension
        base_id = f"{model_name}-{dimension}"
        specs.append(
            EmbeddingSpec(
                id=f"{base_id}-document",
                model_name=model_name,
                dimension=dimension,
                metric=metric,
                normalize=True,
                level="document",
                params={"M": 16, "ef_construction": 200, "ef_search": 50},
            )
        )
        specs.append(
            EmbeddingSpec(
                id=f"{base_id}-chunk",
                model_name=model_name,
                dimension=dimension,
                metric=metric,
                normalize=True,
                level="chunk",
                params={"M": 16, "ef_construction": 200, "ef_search": 50},
            )
        )
    return specs


def _ensure_primary_embedder(
    embedder_models: tuple[EmbedderName, ...],
    embedder: EmbedderName,
) -> tuple[EmbedderName, ...]:
    if embedder in embedder_models:
        return embedder_models
    return (embedder, *embedder_models)


__all__ = ["Container", "ContainerConfig", "build_default_container"]
