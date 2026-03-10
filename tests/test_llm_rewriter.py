import importlib.util
import unittest
from unittest.mock import patch

from domain.entities import Query
from infrastructure.query.llm_rewriter import LLMQueryRewriter, LLMRewriterConfig


@unittest.skipIf(importlib.util.find_spec("transformers") is None, "transformers not installed")
class TestLLMQueryRewriter(unittest.TestCase):
    @patch("transformers.pipeline")
    def test_uses_custom_prompt_template(self, mock_pipeline):
        calls = []

        class _Runner:
            def __call__(self, prompt, **kwargs):
                calls.append(prompt)
                return [{"generated_text": "вариант 1\nвариант 2"}]

        mock_pipeline.return_value = _Runner()
        rewriter = LLMQueryRewriter(
            LLMRewriterConfig(
                model="local/model",
                prompt_template="Перепиши запрос в {count} вариантах: {query}",
                max_queries=2,
            )
        )

        rewritten = rewriter.rewrite(Query(text="как найти документ"))

        self.assertEqual([q.text for q in rewritten], ["вариант 1", "вариант 2"])
        self.assertEqual(calls[0], "Перепиши запрос в 2 вариантах: как найти документ")

    @patch("transformers.pipeline", side_effect=RuntimeError("load error"))
    def test_falls_back_to_original_query_when_model_fails(self, _mock_pipeline):
        rewriter = LLMQueryRewriter(LLMRewriterConfig(model="broken/model"))

        rewritten = rewriter.rewrite(Query(text="оригинал"))

        self.assertEqual([q.text for q in rewritten], ["оригинал"])


if __name__ == "__main__":
    unittest.main()
