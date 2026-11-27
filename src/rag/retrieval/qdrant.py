import json
from collections.abc import Iterable, Iterator
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from ..ingest.chunking import Chunk


CORPUS_ALIAS = "chunks_corpus"
DENSE_VECTOR_NAME = "e5_dense"
DENSE_DIMENSION = 768
BM25_VECTOR_NAME = "bm25_sparse"


class QdrantChunkStore:
    """Store E5 vectors with the metadata needed to find their source."""

    def __init__(self, client: QdrantClient, collection_name: str = "chunks_demo") -> None:
        self._client = client
        self._collection_name = collection_name

    def ensure_collection(self, *, sparse: bool = False) -> None:
        expected = models.VectorParams(size=DENSE_DIMENSION, distance=models.Distance.COSINE)
        if not self._client.collection_exists(self._collection_name):
            sparse_config = (
                {BM25_VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.NONE)}
                if sparse else None
            )
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config={DENSE_VECTOR_NAME: expected},
                sparse_vectors_config=sparse_config,
            )
            return

        params = self._client.get_collection(self._collection_name).config.params
        vectors = params.vectors
        actual = vectors.get(DENSE_VECTOR_NAME) if isinstance(vectors, dict) else None
        if actual is None or actual.size != expected.size or actual.distance != expected.distance:
            raise ValueError(
                f"Collection {self._collection_name!r} requires {DENSE_VECTOR_NAME!r} "
                f"with size {DENSE_DIMENSION} and cosine distance"
            )
        if sparse:
            config = (params.sparse_vectors or {}).get(BM25_VECTOR_NAME)
            if config is None or config.modifier not in (None, models.Modifier.NONE):
                raise ValueError(
                    f"Collection {self._collection_name!r} requires {BM25_VECTOR_NAME!r} "
                    "with IDF supplied by the client"
                )

    def upsert(self, entries: Iterable[tuple[Chunk, list[float]]]) -> None:
        points = []
        for chunk, vector in entries:
            points.append(
                models.PointStruct(
                    id=chunk_point_id(chunk),
                    vector={DENSE_VECTOR_NAME: vector},
                    payload=chunk.to_record(),
                )
            )
        if points:
            self._client.upsert(collection_name=self._collection_name, points=points, wait=True)

    def search(self, query_vector: list[float], limit: int = 5) -> list[models.ScoredPoint]:
        return self._search(query_vector, DENSE_VECTOR_NAME, limit)

    def search_sparse(self, query_vector: models.SparseVector, limit: int) -> list[models.ScoredPoint]:
        if not query_vector.indices:
            return []
        return self._search(query_vector, BM25_VECTOR_NAME, limit)

    def _search(
        self, query_vector: list[float] | models.SparseVector, name: str, limit: int
    ) -> list[models.ScoredPoint]:
        return self._client.query_points(
            collection_name=self._collection_name,
            query=query_vector,
            using=name,
            with_payload=True,
            limit=limit,
        ).points


def iter_points(
    client: QdrantClient,
    collection: str,
    *,
    with_payload: bool = False,
    with_vectors: bool | list[str] = False,
) -> Iterator[models.Record]:
    offset = None
    while True:
        points, offset = client.scroll(
            collection, limit=256, offset=offset,
            with_payload=with_payload, with_vectors=with_vectors,
        )
        yield from points
        if offset is None:
            return


def read_point_ids(client: QdrantClient, collection: str) -> set[str]:
    return {str(point.id) for point in iter_points(client, collection)}


def resolve_collection(client: QdrantClient, name: str) -> str:
    return next(
        (item.collection_name for item in client.get_aliases().aliases if item.alias_name == name),
        name,
    )


def publish_alias(client: QdrantClient, alias: str, collection: str) -> None:
    current = resolve_collection(client, alias)
    if current == collection:
        return

    operations: list[models.DeleteAliasOperation | models.CreateAliasOperation] = []
    if current != alias:
        operations.append(models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias)))
    operations.append(
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(collection_name=collection, alias_name=alias)
        )
    )
    client.update_collection_aliases(change_aliases_operations=operations)


def chunk_point_id(chunk: Chunk) -> str:
    identity_parts: list[object] = [str(chunk.file_path), chunk.source_locator, chunk.order]
    if chunk.source_start_line is not None:
        identity_parts.append(chunk.source_start_line)

    identity = json.dumps(
        identity_parts,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return str(uuid5(NAMESPACE_URL, f"urn:rag:chunk:{identity}"))
