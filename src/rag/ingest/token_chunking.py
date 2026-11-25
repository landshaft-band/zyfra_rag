import re
from bisect import bisect_right
from collections.abc import Iterator
from dataclasses import replace

from tokenizers import Tokenizer

from .chunking import Chunk, split_document_section
from .sources import Source
from .tokenization import PASSAGE_TOKEN_LIMIT, measure_chunk


BODY_TOKEN_LIMIT = 384
OVERLAP_TOKEN_LIMIT = 64


def chunk_source(source: Source, tokenizer: Tokenizer) -> list[Chunk]:
    """Chunk a documentation section or a resolved ticket."""
    if source.kind == "documentation":
        return split_document_section_by_tokens(source, tokenizer)
    if source.kind != "ticket":
        raise ValueError(f"Unsupported source kind: {source.kind!r}")

    block = Chunk(
        file_path=source.file_path,
        source_locator=source.locator,
        order=1,
        heading=source.title,
        text=source.body,
    )
    return _split_blocks([block], tokenizer)


def split_document_section_by_tokens(source: Source, tokenizer: Tokenizer) -> list[Chunk]:
    """Split a section while preserving its subheadings and source references."""
    return _split_blocks(split_document_section(source), tokenizer)


def _split_blocks(blocks: list[Chunk], tokenizer: Tokenizer) -> list[Chunk]:
    if tokenizer.truncation is not None or tokenizer.padding is not None:
        raise ValueError("Chunking requires a tokenizer without truncation or padding")

    chunks: list[Chunk] = []
    for block in blocks:
        for text in _split_block(block, tokenizer):
            chunks.append(replace(block, text=text, order=len(chunks) + 1))
    return chunks


def _fits(chunk: Chunk, tokenizer: Tokenizer) -> bool:
    counts = measure_chunk(chunk, tokenizer)
    return counts.body_tokens <= BODY_TOKEN_LIMIT and counts.passage_tokens <= PASSAGE_TOKEN_LIMIT


def _split_block(chunk: Chunk, tokenizer: Tokenizer) -> Iterator[str]:
    text = chunk.text
    if not text.strip():
        return
    if _fits(chunk, tokenizer):
        yield text
        return

    offsets = tokenizer.encode(text, add_special_tokens=False).offsets
    token_ends = sorted({end for _, end in offsets if end > 0} | {len(text)})
    paragraph_ends = [match.end() for match in re.finditer(r"\n(?:[ \t]*\n)+", text)]
    start = previous_end = 0

    while previous_end < len(text):
        end = _find_end(chunk, tokenizer, token_ends, paragraph_ends, start, previous_end)
        if end is None and start < previous_end:
            # A long heading can leave too little room to retain the overlap.
            start = previous_end
            end = _find_end(chunk, tokenizer, token_ends, paragraph_ends, start, previous_end)
        if end is None:
            raise ValueError(
                f"Cannot fit a fragment of {chunk.source_locator!r} into the E5 limits "
                f"at character {previous_end}"
            )

        yield text[start:end]
        previous_end = end
        if end < len(text):
            start = _overlap_start(text, start, end, tokenizer)


def _find_end(
    chunk: Chunk,
    tokenizer: Tokenizer,
    token_ends: list[int],
    paragraph_ends: list[int],
    start: int,
    previous_end: int,
) -> int | None:
    text = chunk.text
    first_new = bisect_right(token_ends, previous_end)
    last = min(bisect_right(token_ends, start) + BODY_TOKEN_LIMIT, len(token_ends))

    for index in range(last - 1, first_new - 1, -1):
        end = token_ends[index]
        if not _fits(replace(chunk, text=text[start:end]), tokenizer):
            continue
        if end == len(text):
            return end

        paragraph_index = bisect_right(paragraph_ends, end) - 1
        if paragraph_index >= 0:
            paragraph_end = paragraph_ends[paragraph_index]
            paragraph_text = text[start:paragraph_end]
            if paragraph_end > previous_end and paragraph_text.strip():
                if _fits(replace(chunk, text=paragraph_text), tokenizer):
                    return paragraph_end
        return end

    return None


def _overlap_start(text: str, start: int, end: int, tokenizer: Tokenizer) -> int:
    offsets = tokenizer.encode(text[start:end], add_special_tokens=False).offsets
    first = max(1, len(offsets) - OVERLAP_TOKEN_LIMIT)
    for offset, _ in offsets[first:]:
        if offset > 0:
            overlap = tokenizer.encode(text[start + offset:end], add_special_tokens=False)
            if len(overlap.ids) <= OVERLAP_TOKEN_LIMIT:
                return start + offset
    return end
