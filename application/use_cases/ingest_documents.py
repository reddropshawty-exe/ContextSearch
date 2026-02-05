"""Сценарий использования для индексации документов в системе."""
from __future__ import annotations

from hashlib import sha256
from datetime import datetime
from time import time
from typing import Iterable
from uuid import uuid4

from domain.entities import Chunk, Document, EmbeddingSpec
from domain.interfaces import ChunkRepository, ChunkSplitter, DocumentRepository, Embedder, EmbeddingStore, TextExtractor

from application.use_cases.embedding_utils import get_embedder_for_spec


def ingest_documents(
    sources: Iterable[tuple[str | None, bytes | str]],
    *,
    extractor: TextExtractor,
    splitter: ChunkSplitter,
    embedders: dict[str, Embedder],
    embedding_store: EmbeddingStore,
    document_repository: DocumentRepository,
    chunk_repository: ChunkRepository,
    embedding_specs: list[EmbeddingSpec],
) -> list[Document]:
    """Индексировать набор источников, идентифицированных id."""

    ingested_documents: list[Document] = []
    for document_id, source in sources:
        indexed_at = int(time())
        text = extractor.extract(source)
        content_hash = sha256(text.encode("utf-8")).hexdigest()
        document = Document(
            id=document_id or "",
            content=text,
            content_hash=content_hash,
            modified_at=datetime.fromtimestamp(indexed_at),
            metadata={
                "source_type": "raw",
                "source_uri": document_id,
                "display_name": document_id,
                "extension": "",
                "size_bytes": len(text.encode("utf-8")),
                "mtime": indexed_at,
                "content_hash": content_hash,
                "indexed_at": indexed_at,
            },
        )
        document_repository.add(document)
        ingested_documents.append(document)

        chunks = splitter.split(document)
        if not chunks:
            continue
        persisted_chunks: list[Chunk] = []
        for chunk in chunks:
            chunk.metadata = dict(chunk.metadata)
            chunk.text_hash = sha256(chunk.text.encode("utf-8")).hexdigest()
            if not chunk.id:
                chunk.id = str(uuid4())
            chunk_repository.add(chunk)
            persisted_chunks.append(chunk)
        _index_embeddings(
            embedders=embedders,
            embedding_store=embedding_store,
            embedding_specs=embedding_specs,
            document=document,
            chunks=persisted_chunks,
        )

    return ingested_documents


def _index_embeddings(
    *,
    embedders: dict[str, Embedder],
    embedding_store: EmbeddingStore,
    embedding_specs: list[EmbeddingSpec],
    document: Document,
    chunks: list[Chunk],
) -> None:
    for spec in embedding_specs:
        embedder = get_embedder_for_spec(spec, embedders)
        if spec.level == "document":
            doc_embedding = embedder.embed_texts([document.content])[0]
            embedding_store.add(spec, "document", [document.id], [doc_embedding])
        else:
            embeddings = embedder.embed_texts([chunk.text for chunk in chunks])
            embedding_store.add(spec, "chunk", [chunk.id for chunk in chunks], embeddings)
