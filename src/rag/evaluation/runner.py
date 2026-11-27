import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean
from time import perf_counter

from ..generation.client import ChatClient
from ..generation.context import MAX_CONTEXT_CHUNKS, build_context
from ..generation.service import answer_from_context
from ..observability import observation
from ..retrieval.deduplication import RRF_POOL_LIMIT, deduplicate_hits
from ..retrieval.hybrid import BRANCH_LIMIT, HybridRetriever, reciprocal_rank_fusion
from .rerank import RerankSettings, rerank_or_fallback


def load_cases(path: Path, selected: list[str]) -> list[dict]:
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = {case["id"] for case in cases}
    if not cases or len(ids) != len(cases):
        raise ValueError("Нужны контрольные вопросы с уникальными ID.")
    if unknown := set(selected) - ids:
        raise ValueError(f"Неизвестные ID вопросов: {', '.join(sorted(unknown))}")
    return [case for case in cases if not selected or case["id"] in selected]


def _first_rank(points, expected: list[dict]) -> int | None:
    for rank, point in enumerate(points, start=1):
        payload = point.payload or {}
        if any(
            payload.get("source_locator") == source["locator"]
            and ("start_line" not in source or payload.get("source_start_line") == source["start_line"])
            for source in expected
        ):
            return rank
    return None


def evaluate(
    retriever: HybridRetriever,
    cases: Sequence[dict],
    *,
    use_reranker: bool,
    score_answers: bool,
    progress: Callable[[str], None],
) -> list[dict]:
    settings = RerankSettings.from_environment() if use_reranker else None
    usage: list[dict] = []
    model = None
    judge = None
    if score_answers:
        from .ragas_metrics import RagasJudge

        model = replace(ChatClient.from_environment(), on_usage=usage.append, timeout=60)
        judge = RagasJudge(model)

    records = []
    for case in cases:
        progress(case["id"])
        usage.clear()
        with observation("rag.eval.case") as trace:
            question = case["question"]
            start = perf_counter()
            with observation("hybrid-search", "retriever") as search:
                dense, sparse = retriever.search_branches(question, BRANCH_LIMIT)
                raw = reciprocal_rank_fusion(dense, sparse, 20)
                pool = reciprocal_rank_fusion(dense, sparse, RRF_POOL_LIMIT, retriever.search_titles(question))
                unique = deduplicate_hits(pool, max(20, settings.top_n) if settings else 20)
                candidates = unique.hits
                search(metadata={
                    "count": len(candidates), "ids": [str(hit.point.id) for hit in candidates],
                    "exact_duplicates_removed": unique.exact_removed,
                    "near_duplicates_removed": unique.near_removed,
                })
            retrieval_seconds = perf_counter() - start
            expected = case["sources"]
            ranks = {
                "dense": _first_rank(dense, expected),
                "bm25": _first_rank(sparse, expected),
                "rrf_raw": _first_rank([hit.point for hit in raw], expected),
                "rrf": _first_rank([hit.point for hit in candidates[:20]], expected),
            }
            variants = {"rrf": candidates[:MAX_CONTEXT_CHUNKS]}
            record = {
                "id": case["id"], "question": question, "expected_status": case["expected_status"],
                "expected_sources": expected, "reference": case.get("reference"), "ranks": ranks,
                "retrieval_seconds": round(retrieval_seconds, 3),
            }
            if settings is not None:
                result = rerank_or_fallback(question, candidates, settings, on_usage=usage.append)
                if result.fallback_reason is not None:
                    record["rerank_error"] = result.fallback_reason
                record["rerank_seconds"] = round(result.seconds, 3)
                ranks["reranked"] = _first_rank([hit.point for hit in result.hits], expected)
                variants["reranked"] = result.hits

            metrics = {}
            if judge is not None:
                record["answers"] = {
                    name: _evaluate_answer(question, hits, case.get("reference"), model, judge)
                    for name, hits in variants.items()
                }
                metrics = {name: answer.get("ragas", {}) for name, answer in record["answers"].items()}
            trace(metadata={"case_id": case["id"], "ranks": ranks, "ragas": metrics})
        record["usage"] = list(usage)
        records.append(record)
    return records


def _evaluate_answer(question, hits, reference, model, judge) -> dict:
    sources = tuple(build_context(hits))
    start = perf_counter()
    result = answer_from_context(question, sources, model)
    record = asdict(result)
    record["generation_seconds"] = round(perf_counter() - start, 3)
    record["context"] = [{"source_id": source.source_id, "locator": source.locator,
                          "start_line": source.start_line, "text": source.text} for source in sources]
    if result.status == "answer":
        answer = "\n".join(item.text for item in result.items)
        record["ragas"], record["ragas_errors"] = judge.score(
            question, answer, [source.text for source in sources], reference
        )
    return record


def summarize(records: Sequence[dict]) -> dict:
    labeled = [row for row in records if row["expected_sources"]]
    modes = records[0]["ranks"]
    ranks = {}
    for mode in modes:
        positions = [row["ranks"][mode] for row in labeled]
        top = [position for position in positions if position is not None and position <= 8]
        ranks[mode] = {
            "hit_at_8": len(top) / len(positions) if positions else None,
            "mrr_at_8": sum(1 / rank for rank in top) / len(positions) if positions else None,
        }
    usage = [item for row in records for item in row["usage"]]
    result = {
        "cases": len(records), "labeled": len(labeled), "ranking": ranks,
        "rrf_hit_at_20": (
            sum(row["ranks"]["rrf"] is not None for row in labeled) / len(labeled) if labeled else None
        ),
        "rerank_failures": sum("rerank_error" in row for row in records),
        "prompt_tokens": sum(item.get("prompt_tokens", 0) or 0 for item in usage),
        "completion_tokens": sum(item.get("completion_tokens", 0) or 0 for item in usage),
    }
    if usage and all(isinstance(item.get("cost_rub"), (int, float)) for item in usage):
        result["cost_rub"] = round(sum(item["cost_rub"] for item in usage), 4)
    if rerank_times := [row["rerank_seconds"] for row in records if "rerank_seconds" in row]:
        result["mean_rerank_seconds"] = round(mean(rerank_times), 3)
    if "answers" in records[0]:
        result["generation"] = {
            mode: _summarize_answers(records, mode) for mode in records[0]["answers"]
        }
    return result


def _summarize_answers(records: Sequence[dict], mode: str) -> dict:
    answers = [row["answers"][mode] for row in records]
    summary = {
        "status_matches": sum(
            answer["status"] == row["expected_status"]
            for answer, row in zip(answers, records, strict=True)
        ),
        "mean_generation_seconds": round(mean(answer["generation_seconds"] for answer in answers), 3),
    }
    for metric in ("faithfulness", "factual_recall"):
        values = [answer.get("ragas", {}).get(metric) for answer in answers]
        scores = [value for value in values if value is not None]
        summary[metric] = {"mean": mean(scores) if scores else None, "evaluated": len(scores)}
    return summary
