import json
from collections.abc import Callable, Sequence
from hashlib import sha256
from itertools import batched

from qdrant_client import QdrantClient

from ..ingest.chunking import Chunk
from ..ingest.tokenization import E5_MODEL, E5_REVISION, format_passage
from .e5 import E5Embedder
from .qdrant import CORPUS_ALIAS, QdrantChunkStore, chunk_point_id, publish_alias, read_point_ids


def index_corpus(
    client: QdrantClient,
    chunks: Sequence[Chunk],
    *,
    batch_size: int = 8,
    progress: Callable[[int, int], None] | None = None,
) -> str:
    """Build a versioned collection and publish it under the search alias."""
    if not chunks or batch_size < 1:
        raise ValueError("Indexing requires chunks and a positive batch size")
    chunks_by_id = {chunk_point_id(chunk): chunk for chunk in chunks}
    if len(chunks_by_id) != len(chunks):
        raise ValueError("Multiple chunks have the same point ID")

    collection = _collection_name(chunks)
    store = QdrantChunkStore(client, collection)
    store.ensure_collection()
    indexed_ids = read_point_ids(client, collection)
    if indexed_ids - chunks_by_id.keys():
        raise ValueError(f"Collection {collection!r} contains unexpected points")

    missing = [chunk for point_id, chunk in chunks_by_id.items() if point_id not in indexed_ids]
    total = len(chunks)
    if progress is not None:
        progress(len(indexed_ids), total)
    if missing:
        # Similar lengths in a batch keep the padding short.
        missing.sort(key=lambda chunk: len(format_passage(chunk)))
        embedder = E5Embedder()
        done = len(indexed_ids)
        for batch in batched(missing, batch_size):
            vectors = embedder.embed_passages(batch).tolist()
            store.upsert(zip(batch, vectors, strict=True))
            done += len(batch)
            if progress is not None:
                progress(done, total)

    if read_point_ids(client, collection) != chunks_by_id.keys():
        raise RuntimeError(f"Collection {collection!r} is incomplete")
    publish_alias(client, CORPUS_ALIAS, collection)
    return collection


def _collection_name(chunks: Sequence[Chunk]) -> str:
    digest = sha256(f"{E5_MODEL}@{E5_REVISION}\n".encode())
    for chunk in chunks:
        record = json.dumps(chunk.to_record(), ensure_ascii=False, separators=(",", ":"))
        digest.update(record.encode("utf-8"))
        digest.update(b"\n")
    return f"{CORPUS_ALIAS}_{digest.hexdigest()[:16]}"
