from collections.abc import Sequence

import torch
from torch import Tensor
from torch.nn.functional import normalize
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel

from ..ingest.chunking import Chunk
from ..ingest.tokenization import (
    E5_MODEL,
    E5_REVISION,
    PASSAGE_TOKEN_LIMIT,
    format_passage,
    load_e5_tokenizer,
)


class E5Embedder:
    """Encode questions and passages using the pinned multilingual E5 model on CPU."""

    def __init__(self) -> None:
        self._tokenizer = load_e5_tokenizer()
        self._model = AutoModel.from_pretrained(
            E5_MODEL,
            revision=E5_REVISION,
            use_safetensors=True,
        )
        self._model.to("cpu").eval()

    def embed_passage(self, chunk: Chunk) -> Tensor:
        return self.embed_passages([chunk])[0]

    def embed_passages(self, chunks: Sequence[Chunk]) -> Tensor:
        """Return one normalized row per chunk, in the same order."""
        return self._embed_batch([format_passage(chunk) for chunk in chunks])

    def embed_query(self, question: str) -> Tensor:
        question = question.strip()
        if not question:
            raise ValueError("Search question must not be empty")
        return self._embed_batch([f"query: {question}"])[0]

    @torch.inference_mode()
    def _embed_batch(self, texts: list[str]) -> Tensor:
        if not texts:
            return torch.empty((0, self._model.config.hidden_size), dtype=torch.float32, device="cpu")

        encodings = self._tokenizer.encode_batch(texts, add_special_tokens=True)
        longest = max(len(encoding.ids) for encoding in encodings)
        if longest > PASSAGE_TOKEN_LIMIT:
            raise ValueError(f"E5 received {longest} tokens, limit is {PASSAGE_TOKEN_LIMIT}")

        input_ids = pad_sequence(
            [torch.tensor(encoding.ids, device="cpu") for encoding in encodings],
            batch_first=True,
            padding_value=self._model.config.pad_token_id,
        )
        attention_mask = pad_sequence(
            [torch.tensor(encoding.attention_mask, device="cpu") for encoding in encodings],
            batch_first=True,
            padding_value=0,
        )
        hidden_states = self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state

        valid_tokens = attention_mask.unsqueeze(-1).bool()
        masked_states = hidden_states.masked_fill(~valid_tokens, 0)
        mean = masked_states.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        return normalize(mean, p=2, dim=1)
