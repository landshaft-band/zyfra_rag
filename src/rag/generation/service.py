from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

from ..evaluation.rerank import RerankSettings, rerank_or_fallback, reranking_enabled
from ..observability import observation, usage_details
from ..retrieval.hybrid import HybridRetriever
from .answer import AnswerItem, parse_answer
from .client import ChatClient, ModelAPIError
from .context import MAX_CONTEXT_CHUNKS, ContextSource, build_context
from .history import standalone_question


AskStatus = Literal["answer", "clarify", "no_answer", "search_only"]


@dataclass(frozen=True)
class AskResult:
    status: AskStatus
    items: tuple[AnswerItem, ...] = ()
    sources: tuple[ContextSource, ...] = ()
    missing_information: tuple[str, ...] = ()
    message: str | None = None


class AskService:
    def __init__(self, retriever: HybridRetriever) -> None:
        self._retriever = retriever

    def ask(self, question: str, *, history: Sequence[tuple[str, str]] = ()) -> AskResult:
        question = question.strip()
        if not question:
            raise ValueError("Введите непустой вопрос.")

        settings = None
        configuration_error = None
        if reranking_enabled():
            try:
                settings = RerankSettings.from_environment()
            except ValueError as error:
                configuration_error = str(error)
                settings = RerankSettings(MAX_CONTEXT_CHUNKS, MAX_CONTEXT_CHUNKS)

        with observation("rag.ask") as trace:
            if history:
                question = standalone_question(question, history)
            limit = max(settings.top_n, MAX_CONTEXT_CHUNKS) if settings else MAX_CONTEXT_CHUNKS
            with observation("hybrid-search", "retriever") as search:
                hits = self._retriever.search(question, limit)
                search(metadata={"count": len(hits), "ids": [str(hit.point.id) for hit in hits]})
            if settings is not None:
                hits = rerank_or_fallback(
                    question, hits, settings,
                    configuration_error=configuration_error,
                ).hits
            result = answer_from_context(question, tuple(build_context(hits)))
            trace(metadata={
                "status": result.status,
                "rerank_enabled": settings is not None,
            })
            return result


def answer_from_context(
    question: str, sources: tuple[ContextSource, ...], model: ChatClient | None = None
) -> AskResult:
    if not sources:
        return AskResult("no_answer", message="Поиск не нашёл источников для ответа.")
    with observation("answer-generation", "generation") as update:
        try:
            model = model or ChatClient.from_environment()
            update(model=model.model, metadata={"context_count": len(sources)})

            def record_usage(usage: dict) -> None:
                if model.on_usage is not None:
                    model.on_usage(usage)
                update(usage_details=usage_details(usage))

            content = replace(model, on_usage=record_usage).complete(question, sources)
            answer = parse_answer(content, {source.source_id for source in sources})
        except (ModelAPIError, ValueError) as error:
            update(level="WARNING", status_message="search_only")
            return AskResult("search_only", sources=sources, message=str(error))
        update(status_message=answer.status)

    cited_ids = {source_id for item in answer.items for source_id in item.source_ids}
    citations = tuple(source for source in sources if source.source_id in cited_ids)
    message = None
    if answer.status == "no_answer":
        message = "В найденных фрагментах недостаточно сведений для ответа."
    return AskResult(answer.status, answer.items, citations, answer.missing_information, message)
