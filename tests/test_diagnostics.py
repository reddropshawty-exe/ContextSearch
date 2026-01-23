"""Диагностические тесты для локализации падений при поиске."""
from __future__ import annotations

import importlib
import logging
import os
import tempfile
import unittest
from pathlib import Path

from domain.entities import Chunk, Document, Query
from infrastructure.embedding.character_ngram_embedder import CharacterNgramEmbedder
from infrastructure.embedding.mean_word_hash_embedder import MeanWordHashEmbedder
from infrastructure.embedding.minilm_embedder import MiniLMEmbedder
from infrastructure.repositories.sqlite_chunk_repository import SqliteChunkRepository
from infrastructure.storage.faiss_embedding_store import FaissCollection, FaissEmbeddingStore
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


class TestFaissStore(DiagnosticLoggingMixin, unittest.TestCase):
    def test_faiss_search_optional(self) -> None:
        if os.getenv("CONTEXTSEARCH_ENABLE_FAISS"):
            if importlib.util.find_spec("faiss") is None:
                self.skipTest("FAISS не установлен.")

            with tempfile.TemporaryDirectory() as tmp_dir:
                db_path = Path(tmp_dir) / "contextsearch.db"
                collection = FaissCollection(collection_id="test", model_id="hash-minilm-16", dimension=16)
                store = FaissEmbeddingStore(
                    collection=collection,
                    index_root=tmp_dir,
                    db_path=db_path,
                )
                embedder = MiniLMEmbedder(dimension=16)
                document = Document(id="doc-1", content="пример текста")
                chunk = Chunk(id="0", document_id=document.id, text=document.content, metadata={})
                embedding = embedder.embed_texts([chunk.text])[0]
                store.add([chunk], [embedding])
                results = store.search(embedding, top_k=1)
                self.assertEqual(len(results), 1)
        else:
            self.skipTest("Переменная CONTEXTSEARCH_ENABLE_FAISS не установлена.")


class TestInMemoryStore(DiagnosticLoggingMixin, unittest.TestCase):
    def test_in_memory_search(self) -> None:
        store = InMemoryEmbeddingStore()
        embedder = MiniLMEmbedder(dimension=16)
        document = Document(id="doc-1", content="пример текста")
        chunk = Chunk(id="1", document_id=document.id, text=document.content, metadata={})
        embedding = embedder.embed_texts([chunk.text])[0]
        store.add([chunk], [embedding])
        results = store.search(embedding, top_k=1)
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

