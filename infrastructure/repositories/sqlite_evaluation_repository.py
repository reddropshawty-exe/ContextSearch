from __future__ import annotations

import json
import sqlite3
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


class SqliteEvaluationRepository:
    def __init__(self, db_path: str | Path = "contextsearch.db") -> None:
        self._db_path = Path(db_path)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS eval_test_suites (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    document_ids TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS eval_test_cases (
                    id TEXT PRIMARY KEY,
                    suite_id TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    source TEXT NOT NULL,
                    relevance_labels TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    FOREIGN KEY(suite_id) REFERENCES eval_test_suites(id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS eval_experiment_configs (
                    id TEXT PRIMARY KEY,
                    embedding_spec_id TEXT NOT NULL,
                    store_type TEXT NOT NULL,
                    use_bm25 INTEGER NOT NULL,
                    ranking_mode TEXT NOT NULL,
                    query_rewriter TEXT NOT NULL,
                    top_k INTEGER NOT NULL,
                    rrf_k INTEGER NOT NULL,
                    metadata TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS eval_experiment_runs (
                    id TEXT PRIMARY KEY,
                    suite_id TEXT NOT NULL,
                    config_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    aggregate_metrics TEXT NOT NULL,
                    FOREIGN KEY(suite_id) REFERENCES eval_test_suites(id),
                    FOREIGN KEY(config_id) REFERENCES eval_experiment_configs(id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS eval_case_results (
                    run_id TEXT NOT NULL,
                    case_id TEXT NOT NULL,
                    ranked_document_ids TEXT NOT NULL,
                    ranked_chunk_ids TEXT NOT NULL,
                    scores TEXT NOT NULL,
                    score_breakdown TEXT NOT NULL,
                    metrics TEXT NOT NULL,
                    PRIMARY KEY(run_id, case_id),
                    FOREIGN KEY(run_id) REFERENCES eval_experiment_runs(id)
                )
                """
            )

    def upsert_suite(self, suite: TestSuite) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO eval_test_suites (id, name, description, document_ids, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                (suite.id, suite.name, suite.description, json.dumps(suite.document_ids), json.dumps({})),
            )
            for case in suite.test_cases:
                labels = [
                    {"document_id": x.document_id, "grade": x.grade, "chunk_id": x.chunk_id}
                    for x in case.relevance_labels
                ]
                conn.execute(
                    """
                    INSERT OR REPLACE INTO eval_test_cases (id, suite_id, query_text, source, relevance_labels, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        case.id,
                        suite.id,
                        case.query_text,
                        case.source,
                        json.dumps(labels),
                        json.dumps(case.metadata),
                    ),
                )

    def upsert_config(self, config: ExperimentConfig) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO eval_experiment_configs
                (id, embedding_spec_id, store_type, use_bm25, ranking_mode, query_rewriter, top_k, rrf_k, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    config.id,
                    config.embedding_spec_id,
                    config.store_type,
                    int(config.use_bm25),
                    config.ranking_mode,
                    config.query_rewriter,
                    config.top_k,
                    config.rrf_k,
                    json.dumps(config.metadata),
                ),
            )

    def list_suites(self) -> list[TestSuite]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, name, description, document_ids FROM eval_test_suites ORDER BY name"
            ).fetchall()
        return [
            TestSuite(
                id=row[0],
                name=row[1],
                description=row[2],
                document_ids=json.loads(row[3]),
                test_cases=[],
            )
            for row in rows
        ]

    def list_configs(self) -> list[ExperimentConfig]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, embedding_spec_id, store_type, use_bm25, ranking_mode, query_rewriter, top_k, rrf_k, metadata
                FROM eval_experiment_configs
                ORDER BY id DESC
                """
            ).fetchall()
        return [
            ExperimentConfig(
                id=row[0],
                embedding_spec_id=row[1],
                store_type=row[2],
                use_bm25=bool(row[3]),
                ranking_mode=row[4],
                query_rewriter=row[5],
                top_k=int(row[6]),
                rrf_k=int(row[7]),
                metadata=json.loads(row[8]),
            )
            for row in rows
        ]

    def compare_runs(self, left_run_id: str, right_run_id: str) -> dict[str, float]:
        runs = {run.id: run for run in self.list_runs()}
        left = runs.get(left_run_id)
        right = runs.get(right_run_id)
        if left is None or right is None:
            return {}
        keys = set(left.aggregate_metrics.keys()) | set(right.aggregate_metrics.keys())
        return {
            key: float(right.aggregate_metrics.get(key, 0.0) - left.aggregate_metrics.get(key, 0.0))
            for key in sorted(keys)
        }

    def clear_runs(self) -> None:
        """Удалить все сохранённые прогоны и их результаты.

        Используется при изменении индексированной коллекции,
        чтобы не держать устаревшие метрики.
        """
        with self._connect() as conn:
            conn.execute("DELETE FROM eval_case_results")
            conn.execute("DELETE FROM eval_experiment_runs")

    def save_run(self, run: ExperimentRun) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO eval_experiment_runs (id, suite_id, config_id, created_at, aggregate_metrics)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run.id,
                    run.test_suite_id,
                    run.experiment_config_id,
                    run.created_at.isoformat(),
                    json.dumps(run.aggregate_metrics),
                ),
            )
            for case in run.case_results:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO eval_case_results
                    (run_id, case_id, ranked_document_ids, ranked_chunk_ids, scores, score_breakdown, metrics)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run.id,
                        case.test_case_id,
                        json.dumps(case.ranked_document_ids),
                        json.dumps(case.ranked_chunk_ids),
                        json.dumps(case.scores),
                        json.dumps(case.score_breakdown),
                        json.dumps(case.metrics),
                    ),
                )

    def list_runs(self) -> list[ExperimentRun]:
        runs: list[ExperimentRun] = []
        with self._connect() as conn:
            run_rows = conn.execute(
                "SELECT id, suite_id, config_id, created_at, aggregate_metrics FROM eval_experiment_runs ORDER BY created_at DESC"
            ).fetchall()
            for run_row in run_rows:
                case_rows = conn.execute(
                    """
                    SELECT case_id, ranked_document_ids, ranked_chunk_ids, scores, score_breakdown, metrics
                    FROM eval_case_results WHERE run_id = ?
                    """,
                    (run_row[0],),
                ).fetchall()
                case_results = [
                    CaseRunResult(
                        test_case_id=row[0],
                        ranked_document_ids=json.loads(row[1]),
                        ranked_chunk_ids=json.loads(row[2]),
                        scores=[float(x) for x in json.loads(row[3])],
                        score_breakdown=list(json.loads(row[4])),
                        metrics={k: float(v) for k, v in json.loads(row[5]).items()},
                    )
                    for row in case_rows
                ]
                runs.append(
                    ExperimentRun(
                        id=run_row[0],
                        test_suite_id=run_row[1],
                        experiment_config_id=run_row[2],
                        created_at=datetime.fromisoformat(run_row[3]),
                        case_results=case_results,
                        aggregate_metrics={k: float(v) for k, v in json.loads(run_row[4]).items()},
                    )
                )
        return runs

    def get_suite(self, suite_id: str) -> TestSuite | None:
        with self._connect() as conn:
            suite_row = conn.execute(
                "SELECT id, name, description, document_ids FROM eval_test_suites WHERE id = ?",
                (suite_id,),
            ).fetchone()
            if suite_row is None:
                return None
            case_rows = conn.execute(
                "SELECT id, query_text, source, relevance_labels, metadata FROM eval_test_cases WHERE suite_id = ?",
                (suite_id,),
            ).fetchall()

        cases = []
        for row in case_rows:
            labels = [RelevanceLabel(**label) for label in json.loads(row[3])]
            cases.append(
                TestCase(
                    id=row[0],
                    query_text=row[1],
                    source=row[2],
                    relevance_labels=labels,
                    metadata=json.loads(row[4]),
                )
            )

        return TestSuite(
            id=suite_row[0],
            name=suite_row[1],
            description=suite_row[2],
            document_ids=json.loads(suite_row[3]),
            test_cases=cases,
        )
