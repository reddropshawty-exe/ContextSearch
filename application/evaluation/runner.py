from __future__ import annotations

from datetime import datetime
from typing import Callable
from uuid import uuid4

from application.evaluation.metrics import aggregate_mean, hit_at_k, mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k
from application.evaluation.models import CaseRunResult, ExperimentConfig, ExperimentRun, TestSuite


SearchCallable = Callable[[str, int], list[dict[str, object]]]


def run_experiment_suite(
    suite: TestSuite,
    config: ExperimentConfig,
    search_fn: SearchCallable,
) -> ExperimentRun:
    case_results: list[CaseRunResult] = []

    for case in suite.test_cases:
        raw_results = search_fn(case.query_text, config.top_k)
        ranked_doc_ids = [str(item.get("document_id", "")) for item in raw_results if item.get("document_id")]
        ranked_chunk_ids = [str(item.get("chunk_id", "")) for item in raw_results if item.get("chunk_id")]
        scores = [float(item.get("score", 0.0)) for item in raw_results]
        breakdown = [dict(item) for item in raw_results]

        relevant_docs = {label.document_id for label in case.relevance_labels if label.grade > 0}
        gains = {label.document_id: int(label.grade) for label in case.relevance_labels}

        case_metrics = {
            f"precision@{config.top_k}": precision_at_k(ranked_doc_ids, relevant_docs, config.top_k),
            f"recall@{config.top_k}": recall_at_k(ranked_doc_ids, relevant_docs, config.top_k),
            f"hit@{config.top_k}": hit_at_k(ranked_doc_ids, relevant_docs, config.top_k),
            f"mrr@{config.top_k}": mrr_at_k(ranked_doc_ids, relevant_docs, config.top_k),
            f"ndcg@{config.top_k}": ndcg_at_k(ranked_doc_ids, gains, config.top_k),
        }

        case_results.append(
            CaseRunResult(
                test_case_id=case.id,
                ranked_document_ids=ranked_doc_ids,
                ranked_chunk_ids=ranked_chunk_ids,
                scores=scores,
                score_breakdown=breakdown,
                metrics=case_metrics,
            )
        )

    aggregate = aggregate_mean([x.metrics for x in case_results])
    return ExperimentRun(
        id=str(uuid4()),
        test_suite_id=suite.id,
        experiment_config_id=config.id,
        created_at=datetime.utcnow(),
        case_results=case_results,
        aggregate_metrics=aggregate,
    )
