import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from rag.retrieval.deduplication import RRF_POOL_LIMIT
from rag.evaluation.runner import evaluate
from rag.evaluation.rerank import RerankSettings, rerank_or_fallback
from rag.generation.client import ChatClient, ModelAPIError
from rag.generation.service import AskService
from rag.observability import observation
from rag.retrieval.hybrid import HybridHit
from rag.web.app import create_app


NO_ANSWER = '{"status":"no_answer","answer_items":[]}'


class RerankingTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {
            "RAG_LLM_BASE_URL": "https://generation.example/v1",
            "RAG_LLM_MODEL": "writer",
            "RAG_LLM_API_KEY": "writer-secret",
            "RAG_RERANK_ENABLED": "true",
            "LANGFUSE_TRACING_ENABLED": "false",
        }, clear=True)
        env.start()
        self.addCleanup(env.stop)
        self.hits = [
            HybridHit(SimpleNamespace(id=i, payload={
                "heading": str(i), "source_locator": [str(i)],
                "text": f"chunk {i}", "file_path": "test.md",
            }), 0.0, None, None)
            for i in range(1, 22)
        ]
        self.retriever = Mock()
        self.retriever.search.side_effect = lambda question, limit: self.hits[:limit]
        self.retriever.search_branches.return_value = ([hit.point for hit in self.hits], [])
        self.retriever.search_titles.return_value = []
        self.service = AskService(self.retriever)

    def test_rerank_order_and_model_reach_generation(self):
        os.environ.update(RAG_RERANK_MODEL="ranker", RAG_RERANK_TOP_K="2")

        def chat(client, system, content, **kwargs):
            payload = json.loads(content)
            if client.model == "ranker":
                self.assertEqual(len(payload["candidates"]), 20)
                self.assertEqual(payload["count"], 2)
                return '{"ids":["C20","C2"]}'
            self.assertEqual([source["text"] for source in payload["sources"]], ["chunk 20", "chunk 2"])
            return '{"status":"answer","answer_items":[{"text":"ok","source_ids":["S1"]}]}'

        with patch.object(ChatClient, "chat", autospec=True, side_effect=chat):
            with self.assertLogs("rag.evaluation.rerank", level="INFO") as log:
                result = self.service.ask("question")
        self.assertEqual(result.status, "answer")
        self.assertEqual(result.sources[0].title, "20")
        self.retriever.search.assert_called_once_with("question", RRF_POOL_LIMIT)
        self.assertIn("'output_ids': ['20', '2']", log.output[0])
        self.assertNotIn("writer-secret", log.output[0])

    def test_fallback_preserves_eight_rrf_hits_and_generates(self):
        os.environ.update(RAG_RERANK_TOP_N="3", RAG_RERANK_TOP_K="2")
        for bad_response in ("not JSON", '{"ids":["C1","C1"]}', ModelAPIError("API timeout")):
            with self.subTest(bad_response=bad_response):
                with patch.object(ChatClient, "chat", side_effect=[bad_response, NO_ANSWER]) as chat:
                    with self.assertLogs("rag.evaluation.rerank", level="WARNING") as log:
                        self.assertEqual(self.service.ask("question").status, "no_answer")
                content = json.loads(chat.call_args.args[1])
                self.assertEqual([source["title"] for source in content["sources"]], list(map(str, range(1, 9))))
                self.assertIn("fallback_reason", log.output[0])

    def test_search_backfills_twenty_candidates_after_deduplication(self):
        first = self.hits[0]
        self.hits[1].point.payload["text"] = first.point.payload["text"] + "\n"
        with patch.object(ChatClient, "chat", side_effect=[
            '{"ids":["C20","C1","C2","C3","C4","C5","C6","C7"]}', NO_ANSWER,
        ]) as chat:
            self.service.ask("question")
        candidates = json.loads(chat.call_args_list[0].args[1])["candidates"]
        self.assertEqual(len(candidates), 20)
        self.assertEqual(len({candidate["text"].strip() for candidate in candidates}), 20)
        sources = json.loads(chat.call_args_list[1].args[1])["sources"]
        self.assertEqual([source["title"] for source in sources], ["21", "1", "3", "4", "5", "6", "7", "8"])

    def test_post_rerank_check_keeps_selected_unique_hits_only(self):
        self.hits[2].point.payload["text"] = self.hits[1].point.payload["text"]
        with patch.object(ChatClient, "chat", return_value='{"ids":["C3","C2"]}'):
            result = rerank_or_fallback("question", self.hits, RerankSettings(20, 2))
        self.assertEqual([hit.point.id for hit in result.hits], [3])

    def test_empty_rerank_selection_skips_generation_and_asks_for_details(self):
        with patch.object(ChatClient, "chat", return_value='{"ids":[]}') as chat:
            result = self.service.ask("question")
        chat.assert_called_once()
        self.assertEqual(result.status, "clarify")
        self.assertTrue(result.missing_information)

    def test_short_rerank_selection_is_not_backfilled(self):
        with patch.object(ChatClient, "chat", side_effect=['{"ids":["C2"]}', NO_ANSWER]) as chat:
            self.service.ask("question")
        sources = json.loads(chat.call_args.args[1])["sources"]
        self.assertEqual([source["title"] for source in sources], ["2"])

    def test_generation_receives_original_document_request_after_search_rewrite(self):
        original = "выдай в чат положение о ролях"
        search_query = "положение о ролях и полномочиях пользователей"
        responses = [json.dumps({"query": search_query}), '{"ids":["C1"]}', NO_ANSWER]
        with patch.object(ChatClient, "chat", side_effect=responses) as chat:
            self.service.ask(original, history=(("user", "Не вижу назначенную роль"),))
        self.retriever.search.assert_called_once_with(search_query, RRF_POOL_LIMIT)
        self.assertEqual(json.loads(chat.call_args.args[1])["question"], original)

    def test_disabled_and_empty_search_do_not_rerank(self):
        with patch.object(ChatClient, "chat", return_value=NO_ANSWER) as chat:
            os.environ["RAG_RERANK_ENABLED"] = "false"
            self.service.ask("question")
            chat.assert_called_once()
            self.retriever.search.assert_called_once_with("question", RRF_POOL_LIMIT)
            chat.reset_mock()
            os.environ["RAG_RERANK_ENABLED"] = "true"
            self.retriever.search.side_effect = None
            self.retriever.search.return_value = []
            self.assertEqual(self.service.ask("question").status, "no_answer")
            chat.assert_not_called()

    def test_credentials_are_only_inherited_for_same_endpoint(self):
        self.assertEqual(ChatClient.from_environment(rerank=True), ChatClient.from_environment())
        os.environ["RAG_RERANK_BASE_URL"] = "https://ranking.example/v1"
        self.assertIsNone(ChatClient.from_environment(rerank=True).api_key)
        os.environ.update(RAG_RERANK_MODEL="ranker", RAG_RERANK_API_KEY="ranking-secret")
        self.assertEqual(ChatClient.from_environment(rerank=True).api_key, "ranking-secret")
        self.assertEqual(ChatClient.from_environment().api_key, "writer-secret")

    def test_invalid_limits_fall_back_without_calling_ranker(self):
        os.environ["RAG_RERANK_TOP_K"] = "9"
        with patch.object(ChatClient, "chat", return_value=NO_ANSWER) as chat:
            with self.assertLogs("rag.evaluation.rerank", level="WARNING") as log:
                self.assertEqual(self.service.ask("question").status, "no_answer")
        chat.assert_called_once()
        self.retriever.search.assert_called_once_with("question", RRF_POOL_LIMIT)
        self.assertIn("TOP_K", log.output[0])

    def test_web_route_uses_reranked_sources(self):
        os.environ["RAG_RERANK_TOP_K"] = "1"
        answer = '{"status":"answer","answer_items":[{"text":"ok","source_ids":["S1"]}]}'
        with (
            patch("rag.web.app.QdrantClient"),
            patch("rag.web.app.HybridRetriever", return_value=self.retriever),
            patch.object(ChatClient, "chat", side_effect=['{"ids":["C20"]}', answer]),
            TestClient(create_app("http://unused", Path("unused"))) as web,
        ):
            response = web.post("/api/ask", json={"question": "question"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["sources"][0]["title"], "20")

    def test_eval_flag_controls_model_calls_and_preserves_baseline(self):
        os.environ.update(RAG_RERANK_ENABLED="false", RAG_RERANK_TOP_K="1")
        case = {
            "id": "case", "question": "question", "expected_status": "answer",
            "sources": [{"locator": ["20"]}],
        }
        with patch.object(ChatClient, "chat", return_value='{"ids":["C20"]}') as chat:
            baseline = evaluate(self.retriever, [case], use_reranker=False, score_answers=False, progress=Mock())
            chat.assert_not_called()
            comparison = evaluate(self.retriever, [case], use_reranker=True, score_answers=False, progress=Mock())
            chat.assert_called_once()
        self.assertEqual(baseline[0]["ranks"]["rrf"], 20)
        self.assertEqual(comparison[0]["ranks"]["rrf"], 20)
        self.assertEqual(comparison[0]["ranks"]["reranked"], 1)

    def test_trace_contains_only_allowed_metadata(self):
        sdk = Mock()
        manager = sdk.start_as_current_observation.return_value
        manager.__enter__ = Mock(return_value=Mock())
        manager.__exit__ = Mock()
        with patch("rag.observability._client", return_value=sdk):
            responses = ['{"ids":["C8","C7","C6","C5","C4","C3","C2","C1"]}', NO_ANSWER]
            with patch.object(ChatClient, "chat", side_effect=responses):
                self.service.ask("private question")
        names = [call.kwargs["name"] for call in sdk.start_as_current_observation.call_args_list]
        self.assertEqual(names, [
            "guardrails.input", "rag.ask", "hybrid-search", "rerank", "answer-generation", "guardrails.output",
        ])
        updates = str(manager.__enter__.return_value.update.call_args_list)
        for private in ("private question", "chunk 8", "writer-secret"):
            self.assertNotIn(private, updates)

    def test_telemetry_failure_does_not_break_request(self):
        sdk = Mock()
        sdk.start_as_current_observation.side_effect = RuntimeError("private details")
        with patch("rag.observability._client", return_value=sdk):
            with self.assertLogs("rag.observability", level="WARNING") as log:
                with observation("test") as update:
                    update(metadata={"status": "ok"})
        self.assertNotIn("private details", log.output[0])
