from dataclasses import dataclass

from tokenizers import Tokenizer

from .chunking import Chunk


E5_MODEL = "intfloat/multilingual-e5-base"
E5_REVISION = "835193815a3936a24a0ee7dc9e3d48c1fbb19c55"
PASSAGE_TOKEN_LIMIT = 512


@dataclass(frozen=True)
class ChunkTokenCounts:
    body_tokens: int
    passage_tokens: int


def load_e5_tokenizer() -> Tokenizer:
    """Load the pinned tokenizer with truncation and padding disabled."""
    tokenizer = Tokenizer.from_pretrained(E5_MODEL, revision=E5_REVISION)
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer


def format_passage(chunk: Chunk) -> str:
    """Build the input shared by token counting and E5 inference."""
    heading_path = chunk.source_locator
    if chunk.heading is not None:
        heading_path = (*heading_path, chunk.heading)

    headings = " → ".join(heading_path)
    return f"passage: {headings}\n{chunk.text}"


def measure_chunk(chunk: Chunk, tokenizer: Tokenizer) -> ChunkTokenCounts:
    """Count the content and full E5 input using an uncapped tokenizer."""
    passage = format_passage(chunk)
    return ChunkTokenCounts(
        body_tokens=len(tokenizer.encode(chunk.text, add_special_tokens=False).ids),
        passage_tokens=len(tokenizer.encode(passage, add_special_tokens=True).ids),
    )
