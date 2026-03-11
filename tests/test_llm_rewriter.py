import importlib.util
import unittest
from unittest.mock import patch

from domain.entities import Query
from infrastructure.query.llm_rewriter import LLMQueryRewriter, LLMRewriterConfig


@unittest.skipIf(importlib.util.find_spec("transformers") is None, "transformers not installed")
class TestLLMQueryRewriter(unittest.TestCase):
    @patch("transformers.AutoModelForSeq2SeqLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_uses_custom_prompt_template(self, mock_tokenizer_factory, mock_model_factory):
        prompts = []

        class _Tokens(dict):
            def items(self):
                return super().items()

        class _Tokenizer:
            def __call__(self, prompt, **kwargs):
                prompts.append(prompt)
                return _Tokens({"input_ids": [1, 2, 3]})

            def decode(self, _generated, skip_special_tokens=True):
                return "вариант 1\nвариант 2"

        class _Model:
            device = None

            def generate(self, **kwargs):
                return [[10, 11, 12]]

        mock_tokenizer_factory.return_value = _Tokenizer()
        mock_model_factory.return_value = _Model()

        rewriter = LLMQueryRewriter(
            LLMRewriterConfig(
                model="local/model",
                prompt_template="Перепиши запрос в {count} вариантах: {query}",
                max_queries=2,
            )
        )

        rewritten = rewriter.rewrite(Query(text="как найти документ"))

        self.assertEqual([q.text for q in rewritten], ["вариант 1", "вариант 2"])
        self.assertEqual(prompts[0], "Перепиши запрос в 2 вариантах: как найти документ")

    @patch("transformers.AutoTokenizer.from_pretrained", side_effect=RuntimeError("load error"))
    def test_falls_back_to_original_query_when_model_fails(self, _mock_tokenizer_factory):
        rewriter = LLMQueryRewriter(LLMRewriterConfig(model="broken/model"))

        rewritten = rewriter.rewrite(Query(text="оригинал"))

        self.assertEqual([q.text for q in rewritten], ["оригинал"])


if __name__ == "__main__":
    unittest.main()
