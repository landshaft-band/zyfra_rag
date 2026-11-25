import json
import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import UUID, uuid4

import psycopg
from fastapi.testclient import TestClient
from psycopg import sql
from psycopg.conninfo import make_conninfo

from rag.conversations.storage import ConversationStore, migrate
from rag.generation.client import ChatClient, ModelAPIError
from rag.generation.history import REWRITE_PROMPT
from rag.generation.service import AskService
from rag.retrieval.hybrid import HybridHit
from rag.web.app import create_app
from rag.web.conversations import OWNER_COOKIE, owner_digest


class ConversationConfigurationTests(unittest.TestCase):
    def test_web_without_dsn_still_serves_home(self):
        with (
            patch.dict(os.environ, {"RAG_POSTGRES_DSN": ""}),
            patch("rag.web.app.QdrantClient"),
            patch("rag.web.app.HybridRetriever"),
            TestClient(create_app("http://unused", Path("unused"))) as web,
        ):
            self.assertEqual(web.get("/").status_code, 200)
            self.assertEqual(web.post("/api/conversations").status_code, 503)


@unittest.skipUnless(os.getenv("RAG_TEST_POSTGRES_DSN"), "RAG_TEST_POSTGRES_DSN is not set")
class PostgresConversationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base_dsn = os.environ["RAG_TEST_POSTGRES_DSN"]
        schema = f"test_conversations_{uuid4().hex}"
        with psycopg.connect(base_dsn, autocommit=True) as connection:
            connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))

        def cleanup():
            with psycopg.connect(base_dsn, autocommit=True) as connection:
                connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))

        cls.addClassCleanup(cleanup)
        cls.dsn = make_conninfo(base_dsn, options=f"-c search_path={schema}")
        cls.first_migration = migrate(cls.dsn)
        cls.store = ConversationStore(cls.dsn)

    def test_migrations_can_be_reapplied(self):
        self.assertIn("001_conversations.sql", self.first_migration)
        self.assertEqual(migrate(self.dsn), ())

    def test_messages_are_ordered_across_concurrent_writes(self):
        conversation = self.store.create("owner-a")
        with ThreadPoolExecutor(max_workers=4) as pool:
            written = list(pool.map(
                lambda i: self.store.append_message(conversation.id, "owner-a", "user", f"Вопрос {i}"),
                range(8),
            ))
        _, messages = self.store.history(conversation.id, "owner-a")
        self.assertEqual([message.position for message in messages], list(range(1, 9)))
        self.assertEqual({message.content for message in messages}, {message.content for message in written})
        self.assertIsNone(self.store.append_message(conversation.id, "owner-b", "user", "Чужой вопрос"))
        self.assertIsNone(self.store.history(conversation.id, "owner-b"))

    def test_failed_insert_rolls_back_position(self):
        conversation = self.store.create("rollback-owner")
        with self.assertRaises(psycopg.errors.CheckViolation):
            self.store.append_message(conversation.id, "rollback-owner", "user", "Вопрос", {"status": "answer"})
        message = self.store.append_message(conversation.id, "rollback-owner", "user", "Вопрос")
        self.assertEqual(message.position, 1)

    def test_stale_turn_cannot_add_messages(self):
        chat = self.store.create("stale-owner")
        context = self.store.turn_context(chat.id, "stale-owner")
        self.assertTrue(self.store.append_turn(chat.id, "stale-owner", context.next_position, "Q", "A", {}))
        self.assertFalse(self.store.append_turn(chat.id, "stale-owner", context.next_position, "Q2", "A2", {}))
        _, messages = self.store.history(chat.id, "stale-owner")
        self.assertEqual([message.content for message in messages], ["Q", "A"])

    def test_follow_up_uses_own_history_and_persists_citations(self):
        initial = "Где найти события в архиве?"
        follow_up = "А за ночную смену?"
        resolved = "Где найти события в архиве за ночную смену?"
        retriever = Mock()
        retriever.search.return_value = [HybridHit(
            SimpleNamespace(id=1, payload={
                "file_path": "archive.md", "heading": "Архив", "source_locator": ["Архив"],
                "text": "Укажите временные границы смены.",
            }), 0.0, None, None,
        )]

        def chat(model, system, content, **kwargs):
            request = json.loads(content)
            if system == REWRITE_PROMPT:
                self.assertEqual(request["question"], follow_up)
                self.assertEqual(request["history"], [
                    {"role": "user", "content": initial},
                    {"role": "assistant", "content": "Выберите период."},
                ])
                return json.dumps({"query": resolved})
            self.assertEqual(request["question"], resolved if retriever.search.call_count == 2 else initial)
            return '{"status":"answer","answer_items":[{"text":"Выберите период.","source_ids":["S1"]}]}'

        with (
            patch.dict(os.environ, {
                "RAG_POSTGRES_DSN": self.dsn, "RAG_RERANK_ENABLED": "false",
                "RAG_LLM_BASE_URL": "https://mock.example/v1", "RAG_LLM_MODEL": "mock",
            }),
            patch("rag.web.app.QdrantClient"),
            patch("rag.web.app.HybridRetriever", return_value=retriever),
            patch.object(ChatClient, "chat", autospec=True, side_effect=chat) as model,
            TestClient(create_app("http://unused", Path("unused"))) as web,
        ):
            created = web.post("/api/conversations").json()
            url = f"/api/conversations/{created['id']}"
            self.assertEqual(web.post(f"{url}/turns", json={"question": initial}).status_code, 200)
            self.assertEqual(model.call_count, 1)
            self.assertEqual(web.post(f"{url}/turns", json={"question": follow_up}).status_code, 200)
            self.assertEqual(model.call_count, 3)
            retriever.search.assert_called_with(resolved, 8)
            messages = web.get(url).json()["messages"]
            self.assertEqual(messages[2]["content"], follow_up)
            self.assertEqual(messages[3]["result"]["sources"][0]["file_name"], "archive.md")
            with TestClient(create_app("http://unused", Path("unused"))) as stranger:
                self.assertEqual(stranger.post(f"{url}/turns", json={"question": initial}).status_code, 404)
            self.assertEqual(model.call_count, 3)

    def test_history_can_be_reopened_and_remains_private(self):
        with (
            patch.dict(os.environ, {"RAG_POSTGRES_DSN": self.dsn}),
            patch("rag.web.app.QdrantClient"),
            patch("rag.web.app.HybridRetriever"),
        ):
            with TestClient(create_app("http://unused", Path("unused"))) as first:
                self.assertEqual(first.get("/api/conversations").json(), [])
                created = first.post("/api/conversations")
                self.assertEqual(created.status_code, 201)
                self.assertIn("HttpOnly", created.headers["Set-Cookie"])
                conversation = created.json()
                token = first.cookies[OWNER_COOKIE]
                older = UUID(conversation["id"])
                newer = first.post("/api/conversations")
                self.assertNotIn("Set-Cookie", newer.headers)
                self.assertEqual(first.get("/api/conversations").json(), [newer.json(), conversation])

            result = {"status": "answer", "sources": [{"source_id": "S1"}]}
            self.store.append_message(older, owner_digest(token), "user", "Как открыть архив?")
            self.store.append_message(older, owner_digest(token), "assistant", "Выбери период.", result)
            url = f"/api/conversations/{older}"
            with TestClient(create_app("http://unused", Path("unused"))) as stranger:
                self.assertEqual(stranger.get(url).status_code, 404)
                stranger.post("/api/conversations")
                self.assertEqual(stranger.get(url).status_code, 404)
            with TestClient(create_app("http://unused", Path("unused")), cookies={OWNER_COOKIE: token}) as reopened:
                response = reopened.get(url)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers["Cache-Control"], "no-store")
                payload = response.json()
                self.assertEqual(payload["conversation"], conversation)
                self.assertEqual([message["role"] for message in payload["messages"]], ["user", "assistant"])
                self.assertEqual(payload["messages"][1]["result"], result)


class QuestionRewriteTests(unittest.TestCase):
    def test_failure_uses_original_question(self):
        retriever = Mock()
        retriever.search.return_value = []
        for failure in (ModelAPIError("provider unavailable"), "broken JSON"):
            model = ChatClient("https://mock.example/v1", "mock")
            with (
                self.subTest(failure=failure),
                patch.dict(os.environ, {"RAG_RERANK_ENABLED": "false"}),
                patch.object(ChatClient, "from_environment", return_value=model),
                patch.object(ChatClient, "chat", side_effect=[failure]),
            ):
                AskService(retriever).ask("А ночью?", history=(("user", "Покажи журнал"),))
            retriever.search.assert_called_with("А ночью?", 8)
