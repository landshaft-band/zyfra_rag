import io
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rag.generation.client import ChatClient, SYSTEM_PROMPT
from rag.generation.service import AskService
from rag.observability import cost_fields
from rag.retrieval.hybrid import HybridHit


class TraceDetailsTests(unittest.TestCase):
    def test_cost_currency_and_configured_estimate(self):
        with patch.dict(os.environ, {"RAG_MODEL_PRICES_JSON":
                                    '{"example":{"currency":"rub","input_per_million":50,"output_per_million":100}}'}):
            actual = cost_fields("example", {"cost_rub": 0.02, "cost": 0.02})
            self.assertEqual(actual, {"metadata": {"cost_rub": 0.02, "cost_source": "provider"}})
            usd = cost_fields("example", {"cost_usd": 0.03})
            self.assertEqual(usd["cost_details"], {"total": 0.03})
            estimate = cost_fields("example", {"prompt_tokens": 1000, "completion_tokens": 200})
            self.assertAlmostEqual(estimate["metadata"]["estimated_cost_rub"], 0.07)
            self.assertNotIn("cost_details", estimate)
            self.assertEqual(cost_fields("unknown", {"cost": 0.02}), {})

    def test_exact_model_payload_is_opt_in_and_redacts_credentials(self):
        for enabled in ("false", "true"):
            with self.subTest(enabled=enabled):
                result, stages = self._ask(enabled=enabled, answer="Откройте период.")
                self.assertEqual(result.status, "answer")
                generation = stages["answer-generation"]
                self.assertEqual(generation["usage_details"], {"input": 20, "output": 3, "total": 23})
                self.assertEqual(generation["metadata"]["cost_rub"], 0.003)
                if enabled == "true":
                    self.assertEqual(generation["input"][0], {"role": "system", "content": SYSTEM_PROMPT})
                    user = json.loads(generation["input"][1]["content"])
                    self.assertEqual(user["sources"][0]["text"], "Внутренний фрагмент [REDACTED]")
                    self.assertIn("raw_model_output", stages["answer-validation"]["output"])
                else:
                    self.assertTrue(all("input" not in fields and "output" not in fields for fields in stages.values()))
                self.assertNotIn("test-credential", str(stages))

    def test_blocked_raw_answer_is_not_exported(self):
        result, stages = self._ask(enabled="true", answer=SYSTEM_PROMPT + "\nBLOCKED_RESPONSE_MARKER")
        self.assertEqual(result.status, "blocked")
        self.assertNotIn("BLOCKED_RESPONSE_MARKER", str(stages))
        self.assertNotIn("input", stages["guardrails.output"])
        self.assertNotIn("answer-validation", stages)

    def _ask(self, *, enabled, answer):
        stages = {}
        sdk = Mock()

        def start(name, **kwargs):
            manager = Mock()
            active = Mock()
            active.update.side_effect = lambda **fields: stages.setdefault(name, {}).update(fields)
            manager.__enter__ = Mock(return_value=active)
            manager.__exit__ = Mock()
            return manager

        sdk.start_as_current_observation.side_effect = start
        retriever = Mock()
        retriever.search.return_value = [HybridHit(SimpleNamespace(id=1, payload={
            "heading": "Архив", "source_locator": ["Архив"], "file_path": "archive.md",
            "text": "Внутренний фрагмент test-credential",
        }), 0.0, 1, 1)]
        payload = {
            "choices": [{"message": {"content": json.dumps({
                "status": "answer", "answer_items": [{"text": answer, "source_ids": ["S1"]}],
            }, ensure_ascii=False)}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23, "cost_rub": 0.003},
        }
        with (
            patch.dict(os.environ, {
                "RAG_TRACE_CONTENT": enabled, "RAG_GUARDRAILS_MODE": "enforce", "RAG_RERANK_ENABLED": "false",
                "RAG_LLM_API_KEY": "test-credential",
            }),
            patch("rag.observability._client", return_value=sdk),
            patch.object(ChatClient, "from_environment", return_value=ChatClient(
                "https://test.invalid/v1", "example", "test-credential",
            )),
            patch("rag.generation.client.urlopen", return_value=io.BytesIO(json.dumps(payload).encode())),
        ):
            result = AskService(retriever).ask("Как открыть архив?")
        return result, stages
