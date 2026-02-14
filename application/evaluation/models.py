from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class RelevanceLabel:
    document_id: str
    grade: int = 1
    chunk_id: str | None = None


@dataclass(slots=True)
class TestCase:
    id: str
    query_text: str
    source: str = "user"
    relevance_labels: list[RelevanceLabel] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class TestSuite:
    id: str
    name: str
    description: str = ""
    document_ids: list[str] = field(default_factory=list)
    test_cases: list[TestCase] = field(default_factory=list)


@dataclass(slots=True)
class ExperimentConfig:
    id: str
    embedding_spec_id: str
    store_type: str
    use_bm25: bool
    ranking_mode: str
    query_rewriter: str
    top_k: int = 10
    rrf_k: int = 60
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class CaseRunResult:
    test_case_id: str
    ranked_document_ids: list[str]
    ranked_chunk_ids: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    score_breakdown: list[dict[str, float | str | None]] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class ExperimentRun:
    id: str
    test_suite_id: str
    experiment_config_id: str
    created_at: datetime
    case_results: list[CaseRunResult] = field(default_factory=list)
    aggregate_metrics: dict[str, float] = field(default_factory=dict)
