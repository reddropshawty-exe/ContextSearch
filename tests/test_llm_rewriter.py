import importlib.util
import unittest
from unittest.mock import patch

from domain.entities import Query
from infrastructure.query.llm_rewriter import LLMQueryRewriter, LLMRewriterConfig


@unittest.skipIf(importlib.util.find_spec("transformers") is None, "transformers not installed")
class TestLLMQueryRewriter(unittest.TestCase):
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_prompt_and_query_are_passed_as_single_input_line(self, mock_tokenizer_factory, mock_model_factory):
        calls = []

        class _Tokens(dict):
            def items(self):
                return super().items()

        class _Tokenizer:
            def __call__(self, prompt, **kwargs):
                calls.append(prompt)
                return _Tokens({"input_ids": [1, 2, 3]})

            def decode(self, _generated, skip_special_tokens=True):
                # emulate causal LM response containing prompt + completion
                return calls[0] + " новый переписанный запрос"

        class _Model:
            device = None

            def generate(self, **kwargs):
                return [[10, 11, 12]]

        mock_tokenizer_factory.return_value = _Tokenizer()
        mock_model_factory.return_value = _Model()

        rewriter = LLMQueryRewriter(
            LLMRewriterConfig(
                model="local/model",
                prompt_template="Сделай запрос более точным",
                max_query_length=100,
            )
        )

        rewritten = rewriter.rewrite(Query(text="резюме бэкенд разработчика"))

        self.assertEqual([q.text for q in rewritten], ["новый переписанный запрос"])
        self.assertIn("Инструкция: Сделай запрос более точным", calls[0])
        self.assertIn("Запрос: резюме бэкенд разработчика", calls[0])
        self.assertIn("Переписанный запрос:", calls[0])

    @patch("transformers.AutoTokenizer.from_pretrained", side_effect=RuntimeError("load error"))
    def test_falls_back_to_original_query_when_model_fails(self, _mock_tokenizer_factory):
        rewriter = LLMQueryRewriter(LLMRewriterConfig(model="broken/model"))

        rewritten = rewriter.rewrite(Query(text="оригинал"))

        self.assertEqual([q.text for q in rewritten], ["оригинал"])


if __name__ == "__main__":
    unittest.main()
