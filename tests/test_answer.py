import json
import unittest
from unittest.mock import patch

from rag.generation.answer import parse_answer
from rag.generation.client import ChatClient
from rag.generation.context import ContextSource
from rag.generation.service import answer_from_context


class MissingInformationTests(unittest.TestCase):
    def test_malformed_optional_details_keep_answer_and_source(self):
        model = ChatClient("https://example.test/v1", "fake")
        source = ContextSource("S1", "Архив", "Инструкция", "archive.md", ("Архив",), None)
        for details in (None, [""], {"unexpected": "value"}):
            with self.subTest(details=details):
                content = json.dumps({
                    "status": "answer",
                    "answer_items": [{"text": "Выберите период.", "source_ids": ["S1"]}],
                    "missing_information": details,
                })
                with patch.object(ChatClient, "chat", return_value=content):
                    result = answer_from_context("Как открыть архив?", (source,), model)
                self.assertEqual(result.status, "answer")
                self.assertEqual(result.sources, (source,))
                self.assertEqual(result.missing_information, ())

    def test_clarification_accepts_single_text_but_requires_a_real_detail(self):
        content = {"status": "clarify", "answer_items": [], "missing_information": " Уточните период. "}
        self.assertEqual(parse_answer(json.dumps(content), ()).missing_information, ("Уточните период.",))
        content["missing_information"] = [None, " "]
        with self.assertRaisesRegex(ValueError, "укажите, что нужно уточнить"):
            parse_answer(json.dumps(content), ())
