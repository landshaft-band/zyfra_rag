import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from math import log1p
from pathlib import Path
from typing import Self

from qdrant_client.models import SparseVector

from ..ingest.export import write_jsonl


BM25_VERSION = 1
K1 = 1.2
B = 0.75
TERM_PATTERN = re.compile(r"\d+(?:\.\d+)+|[^\W_]+(?:[-_][^\W_]+)*")
DASHES = str.maketrans({char: "-" for char in "‐‑‒–−"})


def tokenize(text: str) -> list[str]:
    """Split on punctuation while keeping codes and dotted versions together."""
    normalized = unicodedata.normalize("NFC", text).casefold().translate(DASHES)
    return TERM_PATTERN.findall(normalized)


class BM25:
    def __init__(self, document_count: int, total_length: int, document_frequency: dict[str, int]) -> None:
        if document_count < 1 or total_length < 1:
            raise ValueError("BM25 requires a corpus with searchable terms")
        self.document_count = document_count
        self.total_length = total_length
        self.average_length = total_length / document_count
        self.document_frequency = dict(sorted(document_frequency.items()))
        self.term_ids = {term: index for index, term in enumerate(self.document_frequency)}

    @classmethod
    def fit(cls, documents: Iterable[Sequence[str]]) -> Self:
        document_count = total_length = 0
        frequencies: Counter[str] = Counter()
        for terms in documents:
            document_count += 1
            total_length += len(terms)
            frequencies.update(set(terms))
        return cls(document_count, total_length, dict(frequencies))

    def encode_document(self, terms: Sequence[str]) -> SparseVector:
        length_factor = K1 * (1 - B + B * len(terms) / self.average_length)
        weights = {
            self.term_ids[term]: frequency * (K1 + 1) / (frequency + length_factor)
            for term, frequency in Counter(terms).items()
            if term in self.term_ids
        }
        return _sparse_vector(weights)

    def encode_query(self, question: str) -> SparseVector:
        weights = {}
        for term in set(tokenize(question)):
            if term in self.term_ids:
                frequency = self.document_frequency[term]
                weights[self.term_ids[term]] = log1p(
                    (self.document_count - frequency + 0.5) / (frequency + 0.5)
                )
        return _sparse_vector(weights)

    def save(self, path: str | Path) -> None:
        record = {
            "version": BM25_VERSION,
            "k1": K1,
            "b": B,
            "document_count": self.document_count,
            "total_length": self.total_length,
            "document_frequency": self.document_frequency,
        }
        write_jsonl(path, (record,))

    @classmethod
    def load(cls, path: str | Path) -> Self:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        if (record["version"], record["k1"], record["b"]) != (BM25_VERSION, K1, B):
            raise ValueError("BM25 statistics have a different format or scoring configuration")
        return cls(record["document_count"], record["total_length"], record["document_frequency"])


def _sparse_vector(weights: dict[int, float]) -> SparseVector:
    indices = sorted(weights)
    return SparseVector(indices=indices, values=[weights[index] for index in indices])
