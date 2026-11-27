import json
from collections.abc import Iterable
from pathlib import Path
from tempfile import NamedTemporaryFile

from .chunking import Chunk
from .sources import Source


def write_sources(path: str | Path, sources: Iterable[Source]) -> None:
    """Write a UTF-8 JSONL snapshot with one source per line."""
    write_jsonl(path, (_source_record(source) for source in sources))


def write_chunks(path: str | Path, chunks: Iterable[Chunk]) -> None:
    """Write a UTF-8 JSONL snapshot with one chunk per line."""
    write_jsonl(path, (chunk.to_record() for chunk in chunks))


def _source_record(source: Source) -> dict[str, object]:
    record: dict[str, object] = {
        "kind": source.kind,
        "file_path": str(source.file_path),
        "locator": source.locator,
        "title": source.title,
        "body": source.body,
    }
    if source.start_line is not None:
        record["start_line"] = source.start_line
    return record


def write_jsonl(path: str | Path, records: Iterable[dict[str, object]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=destination.parent,
        delete=False,
    )
    temporary_path = Path(temporary_file.name)

    try:
        with temporary_file:
            for record in records:
                temporary_file.write(json.dumps(record, ensure_ascii=False) + "\n")

        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)
