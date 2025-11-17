"""Use case for ingesting documents into the system."""
from __future__ import annotations

from typing import Iterable

from domain.entities import Document
from domain.interfaces import (
    ChunkSplitter,
    DocumentRepository,
    Embedder,
    EmbeddingStore,
    TextExtractor,
)


def ingest_documents(
    sources: Iterable[tuple[str, bytes | str]],
    *,
    extractor: TextExtractor,
    splitter: ChunkSplitter,
    embedder: Embedder,
    embedding_store: EmbeddingStore,
    document_repository: DocumentRepository,
) -> list[Document]:
    """Ingest a sequence of sources identified by their ids."""

    ingested_documents: list[Document] = []
    for document_id, source in sources:
        text = extractor.extract(source)
        document = Document(id=document_id, content=text)
        document_repository.add(document)
        ingested_documents.append(document)

        chunks = splitter.split(document)
        if not chunks:
            continue
        embeddings = embedder.embed_texts([chunk.text for chunk in chunks])
        embedding_store.add(chunks, embeddings)

    return ingested_documents
