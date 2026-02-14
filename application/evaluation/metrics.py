from __future__ import annotations

import math


def precision_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = ranked_ids[:k]
    if not top:
        return 0.0
    hit = sum(1 for doc_id in top if doc_id in relevant_ids)
    return hit / len(top)


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top = ranked_ids[:k]
    hit = sum(1 for doc_id in top if doc_id in relevant_ids)
    return hit / len(relevant_ids)


def hit_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    return 1.0 if any(doc_id in relevant_ids for doc_id in ranked_ids[:k]) else 0.0


def mrr_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    for rank, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: list[str], gains: dict[str, int], k: int) -> float:
    if k <= 0:
        return 0.0

    def dcg(ids: list[str]) -> float:
        score = 0.0
        for i, doc_id in enumerate(ids[:k], start=1):
            rel = max(0, gains.get(doc_id, 0))
            if rel <= 0:
                continue
            score += (2**rel - 1) / math.log2(i + 1)
        return score

    observed = dcg(ranked_ids)
    ideal_ids = [doc_id for doc_id, _ in sorted(gains.items(), key=lambda item: item[1], reverse=True)]
    ideal = dcg(ideal_ids)
    if ideal == 0:
        return 0.0
    return observed / ideal


def aggregate_mean(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {}
    keys = set().union(*(m.keys() for m in metrics))
    out: dict[str, float] = {}
    for key in sorted(keys):
        vals = [m[key] for m in metrics if key in m]
        out[key] = sum(vals) / len(vals) if vals else 0.0
    return out
