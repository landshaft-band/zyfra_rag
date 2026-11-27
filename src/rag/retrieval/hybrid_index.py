from hashlib import sha256
from itertools import batched
from pathlib import Path

from qdrant_client import QdrantClient, models

from .bm25 import B, BM25_VERSION, K1, BM25, tokenize
from .qdrant import (
    BM25_VECTOR_NAME,
    CORPUS_ALIAS,
    DENSE_DIMENSION,
    DENSE_VECTOR_NAME,
    QdrantChunkStore,
    iter_points,
    publish_alias,
    read_point_ids,
    resolve_collection,
)


HYBRID_ALIAS = "chunks_hybrid"


def build_hybrid_index(client: QdrantClient, stats_directory: Path) -> tuple[str, int]:
    """Add BM25 to a snapshot of the dense index and publish the new collection."""
    dense_collection = resolve_collection(client, CORPUS_ALIAS)
    if not client.collection_exists(dense_collection):
        raise ValueError("Индекс E5 ещё не собран. Запустите rag build-index.")
    dense_points = list(
        iter_points(client, dense_collection, with_payload=True, with_vectors=[DENSE_VECTOR_NAME])
    )
    documents = [_document_terms(point) for point in dense_points]
    stats = BM25.fit(documents)

    fingerprint = sha256(f"{dense_collection}:{BM25_VERSION}:{K1}:{B}".encode()).hexdigest()
    collection = f"{HYBRID_ALIAS}_{fingerprint[:16]}"
    QdrantChunkStore(client, collection).ensure_collection(sparse=True)
    expected_ids = {str(point.id) for point in dense_points}
    indexed = read_point_ids(client, collection)
    if indexed - expected_ids:
        raise ValueError(f"Collection {collection!r} contains unexpected points")

    missing = (
        _hybrid_point(point, stats.encode_document(terms))
        for point, terms in zip(dense_points, documents, strict=True)
        if str(point.id) not in indexed
    )
    for batch in batched(missing, 64):
        client.upsert(collection, points=list(batch), wait=True)
    if read_point_ids(client, collection) != expected_ids:
        raise RuntimeError(f"Collection {collection!r} is incomplete")

    stats.save(stats_directory / f"{collection}.json")
    publish_alias(client, HYBRID_ALIAS, collection)
    return collection, len(expected_ids)


def _document_terms(point: models.Record) -> list[str]:
    payload = point.payload
    if payload is None:
        raise ValueError(f"Point {point.id!r} has no chunk payload")
    text = "\n".join((*payload["source_locator"], payload["heading"] or "", payload["text"]))
    return tokenize(text)


def _hybrid_point(point: models.Record, sparse: models.SparseVector) -> models.PointStruct:
    dense = point.vector.get(DENSE_VECTOR_NAME) if isinstance(point.vector, dict) else None
    if not isinstance(dense, list) or len(dense) != DENSE_DIMENSION:
        raise ValueError(f"Point {point.id!r} has no compatible E5 vector")
    vectors: dict[str, list[float] | models.SparseVector] = {DENSE_VECTOR_NAME: dense}
    if sparse.indices:
        vectors[BM25_VECTOR_NAME] = sparse
    return models.PointStruct(id=point.id, vector=vectors, payload=point.payload)
