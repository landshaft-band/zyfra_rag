from collections.abc import Sequence
from dataclasses import dataclass

from ..retrieval.hybrid import HybridHit


MAX_CONTEXT_CHUNKS = 8


@dataclass(frozen=True)
class ContextSource:
    source_id: str
    title: str
    text: str
    file_path: str
    locator: tuple[str, ...]
    start_line: int | None

    @property
    def reference(self) -> str:
        if self.start_line is None:
            return self.file_path
        return f"{self.file_path}:{self.start_line}"


def build_context(hits: Sequence[HybridHit]) -> list[ContextSource]:
    sources: list[ContextSource] = []
    for hit in hits[:MAX_CONTEXT_CHUNKS]:
        payload = hit.point.payload
        if not payload or not payload["text"].strip():
            continue
        locator = tuple(payload["source_locator"])
        sources.append(
            ContextSource(
                source_id=f"S{len(sources) + 1}",
                title=payload["heading"] or locator[-1],
                text=payload["text"],
                file_path=payload["file_path"],
                locator=locator,
                start_line=payload.get("source_start_line"),
            )
        )
    return sources
