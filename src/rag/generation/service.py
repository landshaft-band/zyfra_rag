from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal

from ..evaluation.rerank import RerankSettings, rerank_or_fallback, reranking_enabled
from ..guardrails_checks import check_guardrail, guardrail_mode
from ..observability import defer_model_output, observation, trace_content_enabled
from ..retrieval.deduplication import RRF_POOL_LIMIT, deduplicate_hits
from ..retrieval.hybrid import HybridRetriever, is_document_request
from .answer import AnswerItem, parse_answer
from .client import ChatClient, ModelAPIError
from .context import MAX_CONTEXT_CHUNKS, ContextSource, build_context
from .history import standalone_question


AskStatus = Literal["answer", "clarify", "no_answer", "search_only", "blocked"]


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

    def ask(
        self, question: str, *, history: Sequence[tuple[str, str]] = (), input_guard_mode: str | None = None,
    ) -> AskResult:
        question = question.strip()
        if not question:
            raise ValueError("Введите непустой вопрос.")
        mode = guardrail_mode()
        checked_at_submit = input_guard_mode is not None and input_guard_mode == mode
        if not checked_at_submit:
            if notice := check_guardrail("input", question):
                return AskResult("blocked", message=notice)

        settings = None
        configuration_error = None
        if reranking_enabled():
            try:
                settings = RerankSettings.from_environment()
            except ValueError as error:
                configuration_error = str(error)
                settings = RerankSettings(MAX_CONTEXT_CHUNKS, MAX_CONTEXT_CHUNKS)

        with observation("rag.ask") as trace:
            original_question = question
            if history:
                question = standalone_question(question, history)
            limit = max(settings.top_n, MAX_CONTEXT_CHUNKS) if settings else MAX_CONTEXT_CHUNKS
            with observation("hybrid-search", "retriever") as search:
                pool = self._retriever.search(question, RRF_POOL_LIMIT)
                unique = deduplicate_hits(pool, limit)
                hits = unique.hits
                search(metadata={
                    "count": len(pool), "scanned": unique.scanned, "unique_count": len(hits),
                    "exact_duplicates_removed": unique.exact_removed,
                    "near_duplicates_removed": unique.near_removed,
                    "ids": [str(hit.point.id) for hit in hits],
                })
                if trace_content_enabled():
                    search(input=question, output=[
                        {
                            "position": rank, "id": str(hit.point.id), "dense_rank": hit.dense_rank,
                            "bm25_rank": hit.bm25_rank, "text": (hit.point.payload or {}).get("text"),
                            "title_rank": hit.title_rank,
                        }
                        for rank, hit in enumerate(pool[:unique.scanned], start=1)
                    ])
            had_candidates = bool(hits)
            if settings is not None:
                hits = rerank_or_fallback(
                    question, hits, settings,
                    configuration_error=configuration_error,
                ).hits
            with defer_model_output() as raw_output:
                if had_candidates and not hits:
                    result = AskResult("clarify", missing_information=(
                        "Уточните продукт, раздел или название нужного документа.",
                    ))
                else:
                    answer_question = original_question if is_document_request(original_question) else question
                    result = answer_from_context(answer_question, tuple(build_context(hits)))
            output_parts = [item.text for item in result.items] + list(result.missing_information)
            if result.message is not None:
                output_parts.append(result.message)
            if notice := check_guardrail("output", "\n".join(output_parts)):
                trace(metadata={"status": "blocked", "blocked_at": "output"})
                return AskResult("blocked", message=notice)
            if trace_content_enabled():
                with observation("answer-validation") as validation:
                    output = {"status": result.status, "items": [asdict(item) for item in result.items],
                              "missing_information": result.missing_information}
                    if raw_output and result.status != "search_only":
                        output["raw_model_output"] = raw_output[-1]
                    validation(output=output)
            trace(metadata={
                "status": result.status,
                "rerank_enabled": settings is not None,
                "input_guard_at": "submit" if checked_at_submit else "ask",
                "generation_skipped": not hits,
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

            content = model.complete(question, sources)
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
