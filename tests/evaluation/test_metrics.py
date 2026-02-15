import unittest

from application.evaluation.metrics import hit_at_k, mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k


class TestEvaluationMetrics(unittest.TestCase):
    def test_basic_binary_metrics(self):
        ranked = ["d1", "d2", "d3", "d4"]
        rel = {"d2", "d5"}
        self.assertAlmostEqual(precision_at_k(ranked, rel, 2), 0.5)
        self.assertAlmostEqual(recall_at_k(ranked, rel, 2), 0.5)
        self.assertEqual(hit_at_k(ranked, rel, 1), 0.0)
        self.assertEqual(hit_at_k(ranked, rel, 2), 1.0)
        self.assertAlmostEqual(mrr_at_k(ranked, rel, 4), 1 / 2)

    def test_ndcg(self):
        ranked = ["d2", "d1", "d3"]
        gains = {"d1": 3, "d2": 2, "d3": 1}
        score = ndcg_at_k(ranked, gains, 3)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
