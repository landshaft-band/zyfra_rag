import re
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING

from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint

from .bm25 import BM25, tokenize
from .hybrid_index import HYBRID_ALIAS
from .qdrant import QdrantChunkStore, iter_points, resolve_collection

if TYPE_CHECKING:
    from .e5 import E5Embedder


BRANCH_LIMIT = 50
RRF_C = 60
_DOCUMENT_REQUEST = re.compile(r"\b(?:положени\w*|документ\w*|регламент\w*|инструкци\w*|руководств\w*)\b", re.I)
_DOCUMENT_FILLER = {
    "выдай", "найди", "покажи", "пришли", "отправь", "выведи", "предоставь", "дай", "чат", "чате",
    "система", "системе", "пользователь", "пользователя", "название", "нужен",
}


def _title_terms(text: str) -> set[str]:
    return {
        "роль" if re.fullmatch(r"рол(?:ь|и|ей|ям|ях|ями|е|ью)|ролев\w+", word) else word
        for word in tokenize(text)
    }


def is_document_request(question: str) -> bool:
    return bool(_DOCUMENT_REQUEST.search(question))


@dataclass(frozen=True)
class HybridHit:
    point: ScoredPoint
    score: float
    dense_rank: int | None
    bm25_rank: int | None
    title_rank: int | None = None


class HybridRetriever:
    def __init__(self, client: QdrantClient, stats_directory: Path) -> None:
        self._client = client
        self.collection = resolve_collection(client, HYBRID_ALIAS)
        if not client.collection_exists(self.collection):
            raise ValueError("Гибридный индекс ещё не собран. Запустите rag build-hybrid.")
        self._store = QdrantChunkStore(client, self.collection)
        self._stats = BM25.load(stats_directory / f"{self.collection}.json")
        self._embedder: E5Embedder | None = None
        self._embedder_lock = Lock()
        self._title_lock = Lock()
        self._titles: list[tuple[ScoredPoint, set[str], set[str]]] | None = None

    def search_bm25(self, question: str, limit: int) -> list[ScoredPoint]:
        return self._store.search_sparse(self._stats.encode_query(question), limit)

    def search(self, question: str, limit: int) -> list[HybridHit]:
        dense, sparse = self.search_branches(question, BRANCH_LIMIT)
        return reciprocal_rank_fusion(dense, sparse, limit, self.search_titles(question))

    def search_titles(self, question: str) -> list[ScoredPoint]:
        if not is_document_request(question):
            return []
        terms = _title_terms(_DOCUMENT_REQUEST.sub(" ", question)) - _DOCUMENT_FILLER
        terms = {term for term in terms if len(term) >= 4}
        if not terms:
            return []
        with self._title_lock:
            if self._titles is None:
                sections = {}
                for point in iter_points(self._client, self.collection, with_payload=True):
                    payload = point.payload or {}
                    locator = tuple(payload.get("source_locator") or ())
                    heading = payload.get("heading") or ""
                    section = (payload.get("file_path"), locator, heading)
                    if len(locator) < 2 or not payload.get("text"):
                        continue
                    position = (payload.get("order") or 0, payload.get("source_start_line") or 0, str(point.id))
                    if section not in sections or position < sections[section][0]:
                        sections[section] = (position, point)
                self._titles = [
                    (
                        ScoredPoint(id=point.id, version=0, score=0, payload=point.payload),
                        _title_terms(heading), _title_terms(" ".join(locator)),
                    )
                    for (_, locator, heading), (_, point) in sections.items()
                ]
        ranked = []
        for point, heading, path in self._titles:
            title_matches = terms & heading
            path_matches = terms & path
            if title_matches or path_matches:
                ranked.append((len(title_matches), len(path_matches), point))
        ranked.sort(key=lambda row: (-row[0], -row[1], str(row[2].id)))
        return [row[2] for row in ranked[:10]]

    def search_branches(self, question: str, limit: int) -> tuple[list[ScoredPoint], list[ScoredPoint]]:
        with self._embedder_lock:
            if self._embedder is None:
                from .e5 import E5Embedder

                self._embedder = E5Embedder()
            embedder = self._embedder
        vector = embedder.embed_query(question)
        return self._store.search(vector.tolist(), limit=limit), self.search_bm25(question, limit)


def reciprocal_rank_fusion(
    dense: Sequence[ScoredPoint], bm25: Sequence[ScoredPoint], limit: int,
    titles: Sequence[ScoredPoint] = (),
) -> list[HybridHit]:
    """Combine the rankings by point ID using one-based positions."""
    dense_ranks = {point.id: rank for rank, point in enumerate(dense, start=1)}
    bm25_ranks = {point.id: rank for rank, point in enumerate(bm25, start=1)}
    title_ranks = {point.id: rank for rank, point in enumerate(titles, start=1)}
    points = {point.id: point for point in chain(titles, dense, bm25)}

    hits = []
    for point_id, point in points.items():
        dense_rank = dense_ranks.get(point_id)
        bm25_rank = bm25_ranks.get(point_id)
        score = sum(1 / (RRF_C + rank) for rank in (dense_rank, bm25_rank) if rank is not None)
        title_rank = title_ranks.get(point_id)
        if title_rank is not None:
            score += 1.5 / (RRF_C + title_rank)
        hits.append(HybridHit(point, score, dense_rank, bm25_rank, title_rank))

    hits.sort(key=lambda hit: (-hit.score, str(hit.point.id)))
    return hits[:limit]
