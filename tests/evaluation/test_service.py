import importlib.util
import tempfile
import unittest
from pathlib import Path

from infrastructure.repositories.sqlite_evaluation_repository import SqliteEvaluationRepository


@unittest.skipIf(
    importlib.util.find_spec("rank_bm25") is None or importlib.util.find_spec("sentence_transformers") is None,
    "rank_bm25 or sentence_transformers not installed",
)
class TestEvaluationService(unittest.TestCase):
    def test_run_and_save_experiment(self):
        from application.evaluation.service import (
            create_experiment_config,
            create_test_case,
            create_test_suite,
            run_and_save_experiment,
        )
        from application.use_cases.ingest_documents import ingest_documents
        from infrastructure.config import ContainerConfig, build_default_container

        with tempfile.TemporaryDirectory() as tmp:
            container = build_default_container(
                ContainerConfig(
                    data_root=tmp,
                    embedding_store="sqlite",
                    embedder="hash-minilm",
                    embedder_models=("hash-minilm",),
                    safe_mode=False,
                )
            )
            ingest_documents(
                [("doc-1", "alpha beta gamma"), ("doc-2", "delta epsilon")],
                extractor=container.extractor,
                splitter=container.splitter,
                embedders=container.embedders,
                embedding_store=container.embedding_store,
                document_repository=container.document_repository,
                chunk_repository=container.chunk_repository,
                embedding_specs=container.embedding_specs,
                bm25_index=container.bm25_index,
            )

            repo = SqliteEvaluationRepository(db_path=Path(tmp) / "hash-minilm-sqlite" / "contextsearch.db")
            suite = create_test_suite("s1")
            suite.test_cases.append(create_test_case("alpha", [container.document_repository.list()[0].id]))
            cfg = create_experiment_config(
                embedding_spec_id=container.embedding_specs[0].id,
                store_type="sqlite",
                use_bm25=True,
                ranking_mode="hybrid_rrf",
                query_rewriter="none",
                top_k=5,
            )

            run = run_and_save_experiment(suite=suite, config=cfg, container=container, repository=repo)
            self.assertTrue(run.id)
            self.assertTrue(run.aggregate_metrics)
            runs = repo.list_runs()
            self.assertGreaterEqual(len(runs), 1)


if __name__ == "__main__":
    unittest.main()
