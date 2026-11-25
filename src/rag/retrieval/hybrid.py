from collections.abc import Sequence
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING

from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint

from .bm25 import BM25
from .hybrid_index import HYBRID_ALIAS
from .qdrant import QdrantChunkStore, resolve_collection

if TYPE_CHECKING:
    from .e5 import E5Embedder


BRANCH_LIMIT = 50
RRF_C = 60


@dataclass(frozen=True)
class HybridHit:
    point: ScoredPoint
    score: float
    dense_rank: int | None
    bm25_rank: int | None


class HybridRetriever:
    def __init__(self, client: QdrantClient, stats_directory: Path) -> None:
        self.collection = resolve_collection(client, HYBRID_ALIAS)
        if not client.collection_exists(self.collection):
            raise ValueError("Гибридный индекс ещё не собран. Запустите rag build-hybrid.")
        self._store = QdrantChunkStore(client, self.collection)
        self._stats = BM25.load(stats_directory / f"{self.collection}.json")
        self._embedder: E5Embedder | None = None
        self._embedder_lock = Lock()

    def search_bm25(self, question: str, limit: int) -> list[ScoredPoint]:
        return self._store.search_sparse(self._stats.encode_query(question), limit)

    def search(self, question: str, limit: int) -> list[HybridHit]:
        dense, sparse = self.search_branches(question, BRANCH_LIMIT)
        return reciprocal_rank_fusion(dense, sparse, limit)

    def search_branches(self, question: str, limit: int) -> tuple[list[ScoredPoint], list[ScoredPoint]]:
        with self._embedder_lock:
            if self._embedder is None:
                from .e5 import E5Embedder

                self._embedder = E5Embedder()
            embedder = self._embedder
        vector = embedder.embed_query(question)
        return self._store.search(vector.tolist(), limit=limit), self.search_bm25(question, limit)


def reciprocal_rank_fusion(
    dense: Sequence[ScoredPoint], bm25: Sequence[ScoredPoint], limit: int
) -> list[HybridHit]:
    """Combine the rankings by point ID using one-based positions."""
    dense_ranks = {point.id: rank for rank, point in enumerate(dense, start=1)}
    bm25_ranks = {point.id: rank for rank, point in enumerate(bm25, start=1)}
    points = {point.id: point for point in chain(dense, bm25)}

    hits = []
    for point_id, point in points.items():
        dense_rank = dense_ranks.get(point_id)
        bm25_rank = bm25_ranks.get(point_id)
        score = sum(1 / (RRF_C + rank) for rank in (dense_rank, bm25_rank) if rank is not None)
        hits.append(HybridHit(point, score, dense_rank, bm25_rank))

    hits.sort(key=lambda hit: (-hit.score, str(hit.point.id)))
    return hits[:limit]
