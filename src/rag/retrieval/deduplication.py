import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from difflib import SequenceMatcher
from unicodedata import normalize

from .hybrid import BRANCH_LIMIT, HybridHit


RRF_POOL_LIMIT = 2 * BRANCH_LIMIT
_TOKEN = re.compile(r"\w+|[^\w\s]")
_MARKDOWN_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+)$")
_MEANINGFUL_CHANGES = {"не", "нет", "без", "нельзя", "только", "кроме", "если", "запрещено", "разрешено", "+", "-", "−"}


@dataclass(frozen=True)
class UniqueHits:
    hits: list[HybridHit]
    exact_removed: int
    near_removed: int
    scanned: int


def _near_duplicate(first: list[str], second: list[str]) -> bool:
    if min(len(first), len(second)) < 30:
        return False
    matcher = SequenceMatcher(None, first, second, autojunk=False)
    if matcher.real_quick_ratio() < 0.975 or matcher.quick_ratio() < 0.975 or matcher.ratio() < 0.975:
        return False
    for tag, a, b, c, d in matcher.get_opcodes():
        if tag == "equal":
            continue
        changed = (*first[a:b], *second[c:d])
        if _meaningful(changed):
            return False
    return True


def _meaningful(tokens: Iterable[str]) -> bool:
    return any(token.casefold() in _MEANINGFUL_CHANGES or any(char.isdigit() for char in token) for token in tokens)


def _split_headings(text: str) -> tuple[str, set[str]]:
    lines = []
    headings = set()
    for line in text.splitlines():
        heading = _MARKDOWN_HEADING.fullmatch(line)
        if heading is None or _meaningful(_TOKEN.findall(heading[1])):
            lines.append(line)
        else:
            headings.add(" ".join(heading[1].split()))
    return " ".join(" ".join(lines).split()), headings


def deduplicate_hits(hits: Iterable[HybridHit], limit: int) -> UniqueHits:
    """Keep the first ranked representative and its original source metadata."""
    selected: list[HybridHit] = []
    seen: set[str] = set()
    sections: dict[tuple, list[tuple[list[str], str, set[str]]]] = defaultdict(list)
    exact_removed = near_removed = scanned = 0
    for hit in hits:
        if len(selected) >= limit:
            break
        scanned += 1
        payload = hit.point.payload or {}
        raw_text = normalize("NFC", payload.get("text") or "")
        text = " ".join(raw_text.split())
        if not text:
            continue
        if text in seen:
            exact_removed += 1
            continue
        key = (payload.get("file_path"), tuple(payload.get("source_locator") or ()), payload.get("heading"))
        if key[0] and key[1] and key[2]:
            tokens = _TOKEN.findall(text)
            body, headings = _split_headings(raw_text)
            if any(
                (len(tokens) >= 30 and body and body == previous_body
                 and (headings <= previous_headings or previous_headings <= headings))
                or _near_duplicate(tokens, previous)
                for previous, previous_body, previous_headings in sections[key]
            ):
                near_removed += 1
                continue
            sections[key].append((tokens, body, headings))
        selected.append(hit)
        seen.add(text)
    return UniqueHits(selected, exact_removed, near_removed, scanned)
