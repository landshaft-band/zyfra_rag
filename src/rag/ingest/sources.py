import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .documents import Section
from .tickets import Ticket


@dataclass(frozen=True)
class Source:
    kind: Literal["ticket", "documentation"]
    file_path: Path
    locator: tuple[str, ...]
    title: str
    body: str
    start_line: int | None = None


def iter_sources(path: str | Path) -> Iterator[Source]:
    """Read sources from a JSONL export, preserving their source references."""
    with Path(path).open(encoding="utf-8") as source_file:
        for line_number, line in enumerate(source_file, start=1):
            if not line.strip():
                continue
            try:
                source = _parse_source(line)
            except (KeyError, ValueError) as error:
                raise ValueError(f"{path}, line {line_number}: {error}") from error
            yield source


def _parse_source(line: str) -> Source:
    record = json.loads(line)
    if not isinstance(record, dict):
        raise ValueError("expected a JSON object")

    kind = record["kind"]
    if kind not in ("ticket", "documentation"):
        raise ValueError(f"unknown source kind {kind!r}")
    file_path, title, body = record["file_path"], record["title"], record["body"]
    if not all(isinstance(value, str) for value in (file_path, title, body)):
        raise ValueError("file_path, title and body must be strings")
    locator = record["locator"]
    if not isinstance(locator, list) or not locator or not all(isinstance(part, str) for part in locator):
        raise ValueError("locator must be a non-empty list of strings")
    start_line = record.get("start_line")
    if start_line is not None and (type(start_line) is not int or start_line < 1):
        raise ValueError("start_line must be a positive integer")

    return Source(kind, Path(file_path), tuple(locator), title, body, start_line)


def source_from_ticket(ticket: Ticket, file_path: str | Path) -> Source:
    """Represent a ticket as a source with its ID as the locator."""
    return Source(
        kind="ticket",
        file_path=Path(file_path),
        locator=(ticket.ticket_id,),
        title=ticket.title,
        body=ticket.body,
    )


def source_from_section(section: Section, file_path: str | Path) -> Source:
    """Represent a document section as a source with its heading path."""
    return Source(
        kind="documentation",
        file_path=Path(file_path),
        locator=section.heading_path,
        title=section.heading_path[-1],
        body=section.body,
        start_line=section.start_line,
    )
