import logging
from argparse import ArgumentParser, ArgumentTypeError, Namespace
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path


PROCESSED_DATA = Path("data/processed")
BM25_STATS = PROCESSED_DATA / "bm25"
DEFAULT_SOURCES = (
    PROCESSED_DATA / "ticket_sources.jsonl",
    PROCESSED_DATA / "document_sources.jsonl",
)


def main() -> None:
    parser = ArgumentParser(prog="rag", description="Индексировать корпус и искать по нему.")
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    commands = parser.add_subparsers(required=True)

    build = commands.add_parser("build-index", help="Собрать чанки и загрузить векторы в Qdrant.")
    build.add_argument("--sources", nargs="+", type=Path, default=DEFAULT_SOURCES)
    build.add_argument("--output", type=Path, default=PROCESSED_DATA / "corpus_chunks.jsonl")
    build.add_argument("--batch-size", type=_positive_int, default=8)
    build.set_defaults(run=_build_index)

    hybrid = commands.add_parser("build-hybrid", help="Добавить BM25 к сохранённым векторам E5.")
    hybrid.set_defaults(run=_build_hybrid)

    prepare = commands.add_parser("prepare-index", help="Проверить готовность индекса или собрать его из экспортов.")
    prepare.add_argument("--source-directory", type=Path, default=PROCESSED_DATA)
    prepare.set_defaults(run=_prepare_index)

    search = commands.add_parser("search", help="Найти чанки по вопросу.")
    search.add_argument("question", type=_nonempty_question)
    search.add_argument("--limit", type=_positive_int, default=5)
    search.add_argument("--mode", choices=("dense", "bm25", "hybrid"), default="dense")
    search.set_defaults(run=_search)

    ask = commands.add_parser("ask", help="Подготовить ответ по найденным источникам.")
    ask.add_argument("question", type=_nonempty_question)
    ask.set_defaults(run=_ask)

    serve = commands.add_parser("serve", help="Открыть локальный веб-интерфейс помощника.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=_positive_int, default=8000)
    serve.set_defaults(run=_serve)

    migration = commands.add_parser("migrate", help="Применить миграции PostgreSQL.")
    migration.set_defaults(run=_migrate)

    for command, help_text, run in (
        ("queue-init", "Создать топик Kafka и группу Redis Streams.", _queue_init),
        ("queue-publisher", "Публиковать события заданий из PostgreSQL в Kafka.", _queue_publisher),
        ("queue-dispatcher", "Передавать готовые задания из Kafka в Redis.", _queue_dispatcher),
        ("queue-worker", "Выполнять задания из Redis и сохранять ответы.", _queue_worker),
    ):
        commands.add_parser(command, help=help_text).set_defaults(run=run)

    evaluation = commands.add_parser("eval", help="Офлайн-сравнение поиска и проверка ответов.")
    evaluation.add_argument("--questions", type=Path, default=Path("data/eval/questions.jsonl"))
    evaluation.add_argument("--case", action="append", default=[])
    evaluation.add_argument("--output", type=Path)
    evaluation.add_argument("--rerank", action="store_true", help="Сравнить с LLM-реранжированием через API.")
    evaluation.add_argument("--ragas", action="store_true", help="Сгенерировать ответы и оценить их через API.")
    evaluation.set_defaults(run=_evaluate)

    args = parser.parse_args()
    from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

    from .observability import flush

    for name in ("rag.evaluation.rerank", "rag.observability", "rag.jobs"):
        logger = logging.getLogger(name)
        if not logger.handlers:
            logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.INFO)
        logger.propagate = False
    try:
        args.run(args)
    except KeyboardInterrupt:
        pass
    except (OSError, RuntimeError, ValueError, ResponseHandlingException, UnexpectedResponse) as error:
        raise SystemExit(f"Ошибка: {error}") from error
    finally:
        if args.run is not _serve:
            flush()


def _build_index(args: Namespace) -> None:
    from qdrant_client import QdrantClient

    with closing(QdrantClient(url=args.qdrant_url)) as client:
        _index_sources(client, args.sources, args.output, args.batch_size)


def _index_sources(client, sources: Sequence[Path], output: Path, batch_size: int) -> None:
    from .ingest.pipeline import build_corpus
    from .ingest.tokenization import load_e5_tokenizer
    from .retrieval.indexing import CORPUS_ALIAS, index_corpus

    corpus = build_corpus(sources, output, load_e5_tokenizer())
    for kind, count in corpus.source_counts.items():
        print(f"{kind}: {count} источников, {corpus.chunk_counts[kind]} чанков", flush=True)
    for source in corpus.empty_sources:
        print(f"Пустой источник: {source}", flush=True)
    print(f"Чанки: {output.resolve()}", flush=True)

    collection = index_corpus(
        client,
        corpus.chunks,
        batch_size=batch_size,
        progress=lambda done, total: print(f"В Qdrant: {done}/{total} чанков", flush=True),
    )
    print(f"Индекс готов: {CORPUS_ALIAS} → {collection}")


def _build_hybrid(args: Namespace) -> None:
    from qdrant_client import QdrantClient

    from .retrieval.hybrid_index import HYBRID_ALIAS, build_hybrid_index

    with closing(QdrantClient(url=args.qdrant_url)) as client:
        collection, count = build_hybrid_index(client, BM25_STATS)
    print(f"{HYBRID_ALIAS} → {collection}, чанков {count}")


def _prepare_index(args: Namespace) -> None:
    from time import monotonic, sleep

    from qdrant_client import QdrantClient
    from qdrant_client.http.exceptions import ResponseHandlingException

    from .retrieval.bm25 import BM25
    from .retrieval.hybrid_index import HYBRID_ALIAS, build_hybrid_index
    from .retrieval.qdrant import CORPUS_ALIAS, resolve_collection

    deadline = monotonic() + 45
    with closing(QdrantClient(url=args.qdrant_url, check_compatibility=False)) as client:
        while True:
            try:
                hybrid = resolve_collection(client, HYBRID_ALIAS)
                break
            except (OSError, ResponseHandlingException):
                if monotonic() >= deadline:
                    raise
                sleep(1)

        if client.collection_exists(hybrid):
            stats = BM25_STATS / f"{hybrid}.json"
            seed = args.source_directory / "bm25" / stats.name
            if stats.is_file():
                BM25.load(stats)
            elif seed.is_file():
                BM25.load(seed).save(stats)
            if stats.is_file():
                print(f"Индекс уже готов: {HYBRID_ALIAS} → {hybrid}")
                return

        if not client.collection_exists(resolve_collection(client, CORPUS_ALIAS)):
            sources = tuple(args.source_directory / path.name for path in DEFAULT_SOURCES)
            _index_sources(client, sources, PROCESSED_DATA / "corpus_chunks.jsonl", 8)
        collection, count = build_hybrid_index(client, BM25_STATS)
        print(f"{HYBRID_ALIAS} → {collection}, чанков {count}")


def _search(args: Namespace) -> None:
    from qdrant_client import QdrantClient

    from .retrieval.hybrid import HybridRetriever
    from .retrieval.qdrant import CORPUS_ALIAS, QdrantChunkStore, resolve_collection

    with closing(QdrantClient(url=args.qdrant_url)) as client:
        if args.mode == "dense":
            from .retrieval.e5 import E5Embedder

            collection = resolve_collection(client, CORPUS_ALIAS)
            if not client.collection_exists(collection):
                raise ValueError("Индекс ещё не собран. Запустите rag build-index.")
            vector = E5Embedder().embed_query(args.question)
            hits = QdrantChunkStore(client, collection).search(vector.tolist(), limit=args.limit)
            ranked = [(hit, hit.score, "") for hit in hits]
        else:
            retriever = HybridRetriever(client, BM25_STATS)
            if args.mode == "bm25":
                hits = retriever.search_bm25(args.question, args.limit)
                ranked = [(hit, hit.score, "") for hit in hits]
            else:
                ranked = [
                    (hit.point, hit.score, f" · E5 {hit.dense_rank or '—'}, BM25 {hit.bm25_rank or '—'}")
                    for hit in retriever.search(args.question, args.limit)
                ]

    if not ranked:
        print("Совпадений не найдено.")
    for rank, (hit, score, details) in enumerate(ranked, start=1):
        payload = hit.payload or {}
        locator = payload["source_locator"]
        location = " → ".join(locator)
        file_path = payload["file_path"]
        if source_line := payload.get("source_start_line"):
            file_path = f"{file_path}:{source_line}"
        print(f"{rank}. {score:.6f}{details}  {payload['heading'] or locator[-1]}")
        print(f"   {location}\n   {file_path}")
        print(payload["text"], end="\n\n")


def _ask(args: Namespace) -> None:
    from qdrant_client import QdrantClient

    from .generation.service import AskService
    from .retrieval.hybrid import HybridRetriever

    with closing(QdrantClient(url=args.qdrant_url)) as client:
        result = AskService(HybridRetriever(client, BM25_STATS)).ask(args.question)

    if result.status == "search_only":
        print(f"search_only · {result.message}")
        for source in result.sources:
            print(f"\n[{source.source_id}] {source.title} · {' → '.join(source.locator)}")
            print(source.reference)
            print(source.text)
        return

    print(result.status)
    for item in result.items:
        references = " ".join(f"[{ref}]" for ref in item.source_ids)
        print(f"- {item.text} {references}")
    if result.message:
        print(result.message)
    for detail in result.missing_information:
        print(f"- {detail}")
    if result.sources:
        print("\nИсточники")
        for source in result.sources:
            print(f"[{source.source_id}] {' → '.join(source.locator)}")
            print(f"    {source.reference}")


def _serve(args: Namespace) -> None:
    import uvicorn

    from .web.app import create_app

    uvicorn.run(create_app(args.qdrant_url, BM25_STATS), host=args.host, port=args.port)


def _migrate(args: Namespace) -> None:
    import psycopg

    from .conversations.storage import migrate, postgres_dsn

    dsn = postgres_dsn()
    if not dsn:
        raise ValueError("Задайте RAG_POSTGRES_DSN для подключения к PostgreSQL.")
    try:
        applied = migrate(dsn)
    except psycopg.Error as error:
        raise RuntimeError(
            f"Миграции PostgreSQL не применены ({error.sqlstate or type(error).__name__})."
        ) from None
    print(", ".join(applied) if applied else "Миграции PostgreSQL уже применены.")


def _queue_store():
    from .conversations.storage import postgres_dsn
    from .jobs.storage import JobStore

    dsn = postgres_dsn()
    if not dsn:
        raise ValueError("Задайте RAG_POSTGRES_DSN для подключения к PostgreSQL.")
    return JobStore(dsn)


def _queue_init(args: Namespace) -> None:
    from .jobs.runtime import BROKER_ERRORS, QueueSettings, initialize_queue

    try:
        initialize_queue(QueueSettings.from_environment())
    except BROKER_ERRORS as error:
        raise RuntimeError(f"Не удалось подготовить очередь ({type(error).__name__}).") from None
    print("Топик Kafka и группа Redis Streams готовы.")


def _queue_publisher(args: Namespace) -> None:
    from .jobs.runtime import QueueSettings, run_publisher

    run_publisher(_queue_store(), QueueSettings.from_environment())


def _queue_dispatcher(args: Namespace) -> None:
    from .jobs.runtime import QueueSettings, run_dispatcher

    run_dispatcher(QueueSettings.from_environment())


def _queue_worker(args: Namespace) -> None:
    from qdrant_client import QdrantClient

    from .generation.service import AskService
    from .jobs.runtime import QueueSettings, run_worker
    from .retrieval.hybrid import HybridRetriever

    store = _queue_store()
    with closing(QdrantClient(url=args.qdrant_url)) as client:
        service = AskService(HybridRetriever(client, BM25_STATS))
        run_worker(store, service, QueueSettings.from_environment())


def _evaluate(args: Namespace) -> None:
    import json
    import os
    from importlib.util import find_spec

    from qdrant_client import QdrantClient

    from .evaluation.runner import evaluate, load_cases, summarize
    from .ingest.export import write_jsonl
    from .retrieval.hybrid import HybridRetriever

    if args.ragas and find_spec("ragas") is None:
        raise ValueError("Для Ragas запустите uv run --extra evaluation rag eval.")
    if args.ragas:
        os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
    if args.output and args.output.resolve() == args.questions.resolve():
        raise ValueError("Отчёт и контрольный набор должны быть в разных файлах.")
    cases = load_cases(args.questions, args.case)
    with closing(QdrantClient(url=args.qdrant_url)) as client:
        records = evaluate(
            HybridRetriever(client, BM25_STATS), cases, use_reranker=args.rerank,
            score_answers=args.ragas, progress=lambda case: print(case, flush=True),
        )
    if args.output:
        write_jsonl(args.output, records)
    print(json.dumps(summarize(records), ensure_ascii=False, indent=2))


def _nonempty_question(value: str) -> str:
    question = value.strip()
    if not question:
        raise ArgumentTypeError("Введите непустой вопрос.")
    return question


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise ArgumentTypeError("Значение должно быть положительным.")
    return number
