from dataclasses import dataclass
from pathlib import Path

from .sources import Source


@dataclass(frozen=True)
class Chunk:
    file_path: Path
    source_locator: tuple[str, ...]
    order: int
    heading: str | None
    text: str
    source_start_line: int | None = None

    def to_record(self) -> dict[str, object]:
        """Represent the chunk in JSONL and in the Qdrant payload."""
        record: dict[str, object] = {
            "file_path": str(self.file_path),
            "source_locator": list(self.source_locator),
            "order": self.order,
            "heading": self.heading,
            "text": self.text,
        }
        if self.source_start_line is not None:
            record["source_start_line"] = self.source_start_line
        return record


def split_document_section(source: Source) -> list[Chunk]:
    """Split a documentation source into non-empty third-level blocks."""
    if source.kind != "documentation":
        raise ValueError("Expected a documentation source")

    blocks: list[tuple[str | None, list[str]]] = [(None, [])]
    for line in source.body.splitlines():
        if line.startswith(("### ", "###\t")):
            blocks.append((line[3:].strip(), []))
        else:
            blocks[-1][1].append(line)

    chunks: list[Chunk] = []
    for heading, lines in blocks:
        text = "\n".join(lines).strip("\n")
        if not text.strip():
            continue

        chunks.append(
            Chunk(
                file_path=source.file_path,
                source_locator=source.locator,
                order=len(chunks) + 1,
                heading=heading,
                text=text,
                source_start_line=source.start_line,
            )
        )

    return chunks
