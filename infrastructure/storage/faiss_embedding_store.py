"""FAISS-based embedding store with persistence and collections."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import faiss
import numpy as np

from domain.entities import Chunk, RetrievalResult
from domain.interfaces import EmbeddingStore
from infrastructure.repositories.sqlite_chunk_repository import SqliteChunkRepository


@dataclass(slots=True)
class FaissCollection:
    collection_id: str
    model_id: str
    dimension: int
    metric: str = "cosine"
    index_path: str = "faiss.index"
    meta_path: str = "collection.json"


class FaissEmbeddingStore(EmbeddingStore):
    """FAISS-backed embedding store that persists vectors to disk."""

    def __init__(
        self,
        *,
        collection: FaissCollection,
        index_root: str | Path = "indexes",
        db_path: str | Path = "contextsearch.db",
        normalize_embeddings: bool = True,
    ) -> None:
        self.collection_id = collection.collection_id
        self._collection = collection
        self._index_root = Path(index_root)
        self._collection_dir = self._index_root / collection.collection_id
        self._collection_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._collection_dir / collection.index_path
        self._meta_path = self._collection_dir / collection.meta_path
        self._normalize_embeddings = normalize_embeddings
        self._chunk_repository = SqliteChunkRepository(db_path=db_path)
        self._index = self._load_or_create_index()

    def _load_or_create_index(self) -> faiss.IndexIDMap2:
        if self._meta_path.exists():
            stored = json.loads(self._meta_path.read_text(encoding="utf-8"))
            if stored.get("dimension") != self._collection.dimension:
                raise ValueError("FAISS collection dimension mismatch.")
            if stored.get("model_id") != self._collection.model_id:
                raise ValueError("FAISS collection model mismatch.")
        else:
            self._meta_path.write_text(json.dumps(asdict(self._collection), indent=2), encoding="utf-8")

        if self._index_path.exists():
            index = faiss.read_index(str(self._index_path))
            if not isinstance(index, faiss.IndexIDMap2):
                index = faiss.IndexIDMap2(index)
        else:
            base = faiss.IndexFlatIP(self._collection.dimension)
            index = faiss.IndexIDMap2(base)
            faiss.write_index(index, str(self._index_path))
        return index

    def _persist(self) -> None:
        faiss.write_index(self._index, str(self._index_path))

    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        if not chunks:
            return
        if any(len(vector) != self._collection.dimension for vector in embeddings):
            raise ValueError("Embedding dimension does not match FAISS collection.")

        vectors = np.array(embeddings, dtype="float32")
        if self._normalize_embeddings:
            faiss.normalize_L2(vectors)

        ids: list[int] = []
        for chunk in chunks:
            chunk.metadata = dict(chunk.metadata)
            chunk.metadata.setdefault("collection_id", self.collection_id)
            chunk_id = self._chunk_repository.add(chunk)
            chunk.id = str(chunk_id)
            ids.append(chunk_id)

        id_array = np.array(ids, dtype="int64")
        self._index.add_with_ids(vectors, id_array)
        self._persist()

    def search(self, query_embedding: Sequence[float], top_k: int = 5) -> list[RetrievalResult]:
        if self._index.ntotal == 0:
            return []
        vector = np.array([query_embedding], dtype="float32")
        if self._normalize_embeddings:
            faiss.normalize_L2(vector)
        scores, ids = self._index.search(vector, top_k)
        results: list[RetrievalResult] = []
        for score, chunk_id in zip(scores[0], ids[0]):
            if chunk_id < 0:
                continue
            chunk = self._chunk_repository.get(int(chunk_id))
            if chunk is None:
                continue
            results.append(RetrievalResult(chunk=chunk, score=float(score)))
        return results


__all__ = ["FaissEmbeddingStore", "FaissCollection"]
