import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


CHAPTER_OR_SECTION_HEADING = re.compile(
    r"^(?P<level>#{1,2})[ \t]+(?P<title>.+)$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class Section:
    heading_path: tuple[str, str]
    body: str
    start_line: int


def iter_sections(path: str | Path) -> Iterator[Section]:
    """Yield second-level sections and their original heading line numbers."""
    markdown = Path(path).read_text(encoding="utf-8")
    headings = CHAPTER_OR_SECTION_HEADING.finditer(markdown)
    heading = next(headings, None)
    current_chapter = None
    previous_start = 0
    line_number = 1

    while heading is not None:
        line_number += markdown.count("\n", previous_start, heading.start())
        previous_start = heading.start()
        next_heading = next(headings, None)
        title = heading["title"].strip()

        if heading["level"] == "#":
            current_chapter = title
        elif current_chapter is not None:
            body_end = next_heading.start() if next_heading is not None else len(markdown)
            yield Section(
                heading_path=(current_chapter, title),
                body=markdown[heading.end():body_end].strip("\n"),
                start_line=line_number,
            )
        heading = next_heading


def read_section(
    path: str | Path,
    chapter_title: str,
    section_title: str,
    *,
    start_line: int | None = None,
) -> Section:
    """Select a section by its heading path and optional source line."""
    heading_path = (chapter_title, section_title)
    matches = [
        section
        for section in iter_sections(path)
        if section.heading_path == heading_path
        and (start_line is None or section.start_line == start_line)
    ]
    if not matches:
        location = f" at line {start_line}" if start_line is not None else ""
        raise ValueError(f"Section {heading_path!r}{location} was not found in {path}")
    if len(matches) > 1:
        raise ValueError(f"Section {heading_path!r} is ambiguous in {path}")
    return matches[0]
