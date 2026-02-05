"""Диагностические тесты для локализации падений при поиске."""
from __future__ import annotations

import importlib
import logging
import os
import tempfile
import unittest
from pathlib import Path

from domain.entities import Chunk, Document, EmbeddingSpec, Query
from infrastructure.embedding.character_ngram_embedder import CharacterNgramEmbedder
from infrastructure.embedding.mean_word_hash_embedder import MeanWordHashEmbedder
from infrastructure.embedding.minilm_embedder import MiniLMEmbedder
from infrastructure.repositories.sqlite_chunk_repository import SqliteChunkRepository
from infrastructure.repositories.sqlite_embedding_record_repository import SqliteEmbeddingRecordRepository
from infrastructure.storage.hnsw_embedding_store import HnswEmbeddingStore
from infrastructure.storage.in_memory_embedding_store import InMemoryEmbeddingStore


class DiagnosticLoggingMixin:
    def setUp(self) -> None:
        logging.basicConfig(level=logging.DEBUG)


class TestHashEmbedders(DiagnosticLoggingMixin, unittest.TestCase):
    def test_hash_embedders_return_consistent_dimensions(self) -> None:
        embedders = [MiniLMEmbedder(), MeanWordHashEmbedder(), CharacterNgramEmbedder()]
        for embedder in embedders:
            vectors = embedder.embed_texts(["тестовый текст", "ещё один текст"])
            self.assertEqual(len(vectors), 2)
            self.assertEqual(len(vectors[0]), embedder.dimension)
            self.assertEqual(len(vectors[1]), embedder.dimension)


class TestSentenceTransformers(DiagnosticLoggingMixin, unittest.TestCase):
    def test_sentence_transformers_embedder_optional(self) -> None:
        if os.getenv("CONTEXTSEARCH_ENABLE_ST"):
            from infrastructure.embedding.sentence_transformers_embedder import (  # noqa: PLC0415
                SentenceTransformersConfig,
                SentenceTransformersEmbedder,
            )

            embedder = SentenceTransformersEmbedder(SentenceTransformersConfig(model_name="sentence-transformers/all-MiniLM-L6-v2"))
            vector = embedder.embed_query(Query(text="тестовый запрос"))
            self.assertEqual(len(vector), embedder.dimension)
        else:
            self.skipTest("Переменная CONTEXTSEARCH_ENABLE_ST не установлена.")


class TestHnswStore(DiagnosticLoggingMixin, unittest.TestCase):
    def test_hnsw_search_optional(self) -> None:
        if os.getenv("CONTEXTSEARCH_ENABLE_HNSW"):
            if importlib.util.find_spec("hnswlib") is None:
                self.skipTest("hnswlib не установлен.")

            with tempfile.TemporaryDirectory() as tmp_dir:
                db_path = Path(tmp_dir) / "contextsearch.db"
                record_repo = SqliteEmbeddingRecordRepository(db_path=db_path)
                store = HnswEmbeddingStore(index_root=tmp_dir, record_repository=record_repo)
                embedder = MiniLMEmbedder(dimension=16)
                spec = EmbeddingSpec(
                    id="test-doc",
                    model_name="hash-minilm",
                    dimension=16,
                    metric="cosine",
                    normalize=True,
                    level="document",
                )
                document = Document(id="doc-1", content="пример текста")
                embedding = embedder.embed_texts([document.content])[0]
                store.add(spec, "document", [document.id], [embedding])
                results = store.search(spec, embedding, top_k=1)
                self.assertEqual(len(results), 1)
        else:
            self.skipTest("Переменная CONTEXTSEARCH_ENABLE_HNSW не установлена.")


class TestInMemoryStore(DiagnosticLoggingMixin, unittest.TestCase):
    def test_in_memory_search(self) -> None:
        store = InMemoryEmbeddingStore()
        embedder = MiniLMEmbedder(dimension=16)
        spec = EmbeddingSpec(
            id="test-doc",
            model_name="hash-minilm",
            dimension=16,
            metric="cosine",
            normalize=True,
            level="document",
        )
        document = Document(id="doc-1", content="пример текста")
        embedding = embedder.embed_texts([document.content])[0]
        store.add(spec, "document", [document.id], [embedding])
        results = store.search(spec, embedding, top_k=1)
        self.assertEqual(len(results), 1)


class TestChunkRepository(DiagnosticLoggingMixin, unittest.TestCase):
    def test_sqlite_chunk_repository_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "contextsearch.db"
            repo = SqliteChunkRepository(db_path=db_path)
            chunk_id = repo.add(Chunk(id="0", document_id="doc-1", text="пример", metadata={"source_uri": "file:///tmp"}))
            chunk = repo.get(chunk_id)
            self.assertIsNotNone(chunk)
            assert chunk is not None
            self.assertEqual(chunk.text, "пример")
