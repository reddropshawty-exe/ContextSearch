import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from application.evaluation.models import (
    CaseRunResult,
    ExperimentConfig,
    ExperimentRun,
    RelevanceLabel,
    TestCase,
    TestSuite,
)
from infrastructure.repositories.sqlite_evaluation_repository import SqliteEvaluationRepository


class TestSqliteEvaluationRepository(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "eval.db"
            repo = SqliteEvaluationRepository(db_path=db_path)

            suite = TestSuite(
                id="suite-1",
                name="Demo",
                document_ids=["d1", "d2"],
                test_cases=[
                    TestCase(
                        id="case-1",
                        query_text="test query",
                        relevance_labels=[RelevanceLabel(document_id="d1", grade=1)],
                    )
                ],
            )
            repo.upsert_suite(suite)

            cfg = ExperimentConfig(
                id="cfg-1",
                embedding_spec_id="all-minilm-384-document",
                store_type="sqlite",
                use_bm25=True,
                ranking_mode="hybrid_rrf",
                query_rewriter="none",
            )
            repo.upsert_config(cfg)

            run = ExperimentRun(
                id="run-1",
                test_suite_id=suite.id,
                experiment_config_id=cfg.id,
                created_at=datetime.utcnow(),
                case_results=[
                    CaseRunResult(
                        test_case_id="case-1",
                        ranked_document_ids=["d1", "d2"],
                        metrics={"precision@10": 0.5},
                    )
                ],
                aggregate_metrics={"precision@10": 0.5},
            )
            repo.save_run(run)

            loaded_suite = repo.get_suite("suite-1")
            self.assertIsNotNone(loaded_suite)
            assert loaded_suite is not None
            self.assertEqual(len(loaded_suite.test_cases), 1)

            loaded_runs = repo.list_runs()
            self.assertEqual(len(loaded_runs), 1)
            self.assertEqual(loaded_runs[0].id, "run-1")

            suites = repo.list_suites()
            self.assertEqual(len(suites), 1)
            self.assertEqual(suites[0].id, "suite-1")

            configs = repo.list_configs()
            self.assertEqual(len(configs), 1)
            self.assertEqual(configs[0].id, "cfg-1")

            # second run for diff
            run2 = ExperimentRun(
                id="run-2",
                test_suite_id=suite.id,
                experiment_config_id=cfg.id,
                created_at=datetime.utcnow(),
                case_results=[
                    CaseRunResult(
                        test_case_id="case-1",
                        ranked_document_ids=["d1"],
                        metrics={"precision@10": 1.0},
                    )
                ],
                aggregate_metrics={"precision@10": 1.0},
            )
            repo.save_run(run2)
            diff = repo.compare_runs("run-1", "run-2")
            self.assertAlmostEqual(diff.get("precision@10", 0.0), 0.5)

            repo.clear_runs()
            self.assertEqual(len(repo.list_runs()), 0)


if __name__ == "__main__":
    unittest.main()
