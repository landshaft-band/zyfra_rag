import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from qdrant_client.models import Record

from rag.retrieval.hybrid import HybridRetriever


class TitleRetrievalTests(unittest.TestCase):
    def test_document_request_surfaces_role_matrix_by_heading(self):
        matrix = Record(id=1, payload={
            "heading": "Рекомендуемая матрица ролей мониторинга",
            "file_path": "reference.md", "source_locator": ["MDC", "Мониторинг"],
            "text": "Оператор видит состояние оборудования.",
        })
        unrelated = Record(id=2, payload={
            "heading": "Вибродиагностика",
            "file_path": "reference.md", "source_locator": ["MDC", "Мониторинг"],
            "text": "Измерения вибрации.",
        })
        with (
            TemporaryDirectory() as directory,
            patch("rag.retrieval.hybrid.resolve_collection", return_value="test"),
            patch("rag.retrieval.hybrid.BM25.load"),
            patch("rag.retrieval.hybrid.iter_points", return_value=iter([unrelated, matrix])) as points,
        ):
            retriever = HybridRetriever(Mock(), Path(directory))
            with patch.object(retriever, "search_branches", return_value=([], [])):
                self.assertEqual(retriever.search("Как проверить роль?", 20), [])
                points.assert_not_called()
                hits = retriever.search("выдай в чат положение о ролях", 20)
        self.assertEqual([(hit.point.id, hit.title_rank) for hit in hits], [(1, 1)])
