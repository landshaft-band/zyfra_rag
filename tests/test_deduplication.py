import unittest
from types import SimpleNamespace

from rag.retrieval.deduplication import deduplicate_hits
from rag.retrieval.hybrid import HybridHit


def hit(identifier, text, section="Роли"):
    point = SimpleNamespace(id=identifier, payload={
        "file_path": "reference.md",
        "source_locator": ["Инструкция", section],
        "source_start_line": identifier * 10,
        "heading": section,
        "text": text,
    })
    return HybridHit(point, 0.0, None, None)


class DeduplicationTests(unittest.TestCase):
    def test_exact_and_near_copies_keep_original_metadata(self):
        text = " ".join(f"параметр_{chr(1072 + index)}" for index in range(90))
        original = hit(1, text)
        result = deduplicate_hits([
            original,
            hit(2, text.replace(" ", "\n"), "Другой раздел"),
            hit(3, text + "\n#### Примечание"),
            hit(4, text + " не"),
            hit(5, text + " 15"),
            hit(6, text + "\n#### Примечание", "Другой раздел"),
            hit(7, " ".join(text.split()[30:])),
        ], 8)
        self.assertEqual([item.point.id for item in result.hits], [1, 4, 5, 6, 7])
        self.assertIs(result.hits[0], original)
        self.assertEqual(result.hits[0].point.payload["source_start_line"], 10)
        self.assertEqual((result.exact_removed, result.near_removed), (1, 1))

    def test_repeated_body_with_inserted_markdown_heading(self):
        text = " ".join(["доступ", "архив", "акт", "роль", "отчёт"] * 10)
        heading = "\n#### Технические детали и обработка обращений\n"
        result = deduplicate_hits([
            hit(1, text), hit(2, heading + text), hit(3, "\n#### Только администратор\n" + text),
        ], 8)
        self.assertEqual([item.point.id for item in result.hits], [1, 3])
        self.assertEqual(result.near_removed, 1)
