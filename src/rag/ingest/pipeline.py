from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer

from .chunking import Chunk
from .export import write_chunks
from .sources import iter_sources
from .token_chunking import chunk_source


@dataclass(frozen=True)
class CorpusBuild:
    chunks: list[Chunk]
    source_counts: Counter[str]
    chunk_counts: Counter[str]
    empty_sources: list[str]


def build_corpus(
    source_paths: Iterable[str | Path],
    output_path: str | Path,
    tokenizer: Tokenizer,
) -> CorpusBuild:
    """Build and export chunks from one or more source snapshots."""
    chunks: list[Chunk] = []
    source_counts: Counter[str] = Counter()
    chunk_counts: Counter[str] = Counter()
    empty_sources: list[str] = []
    seen_sources: set[tuple[Path, tuple[str, ...], int | None]] = set()
    destination = Path(output_path).resolve()

    for path in source_paths:
        if Path(path).resolve() == destination:
            raise ValueError(f"Chunk output must differ from source file {path}")
        for source in iter_sources(path):
            key = (source.file_path, source.locator, source.start_line)
            if key in seen_sources:
                raise ValueError(f"Duplicate source: {key!r}")
            seen_sources.add(key)

            parts = chunk_source(source, tokenizer)
            source_counts[source.kind] += 1
            chunk_counts[source.kind] += len(parts)
            chunks.extend(parts)
            if not parts:
                location = " → ".join(source.locator)
                empty_sources.append(f"{source.file_path}: {location} (line {source.start_line})")

    if not chunks:
        raise ValueError("The sources contain no indexable chunks")
    write_chunks(output_path, chunks)
    return CorpusBuild(chunks, source_counts, chunk_counts, empty_sources)
