import importlib.util
import unittest


@unittest.skipIf(importlib.util.find_spec("rank_bm25") is None, "rank_bm25 not installed")
class TestSearchRankingModes(unittest.TestCase):
    def test_combsum_prefers_strong_combined_signal(self):
        from application.use_cases.search import _merge_scores

        vector_scores = {"doc-1": 0.9, "doc-2": 0.7}
        bm25_scores = {"doc-1": 0.1, "doc-2": 10.0}

        merged = _merge_scores(vector_scores, bm25_scores, ranking_mode="combsum")

        self.assertGreater(merged["doc-2"], merged["doc-1"])

    def test_combmnz_rewards_documents_present_in_both_rankers(self):
        from application.use_cases.search import _merge_scores

        vector_scores = {"doc-1": 0.8, "doc-2": 0.7}
        bm25_scores = {"doc-1": 2.0}

        combsum = _merge_scores(vector_scores, bm25_scores, ranking_mode="combsum")
        combmnz = _merge_scores(vector_scores, bm25_scores, ranking_mode="combmnz")

        self.assertGreater(combmnz["doc-1"], combsum["doc-1"])
        self.assertGreater(combmnz["doc-1"], combmnz["doc-2"])

    def test_combsum_honors_ranker_weights(self):
        from application.use_cases.search import _merge_scores

        vector_scores = {"doc-1": 0.9, "doc-2": 0.2}
        bm25_scores = {"doc-1": 0.1, "doc-2": 9.0}

        vector_dominant = _merge_scores(
            vector_scores,
            bm25_scores,
            ranking_mode="combsum",
            vector_weight=2.0,
            bm25_weight=0.1,
        )
        bm25_dominant = _merge_scores(
            vector_scores,
            bm25_scores,
            ranking_mode="combsum",
            vector_weight=0.1,
            bm25_weight=2.0,
        )

        self.assertGreater(vector_dominant["doc-1"], vector_dominant["doc-2"])
        self.assertGreater(bm25_dominant["doc-2"], bm25_dominant["doc-1"])


if __name__ == "__main__":
    unittest.main()
