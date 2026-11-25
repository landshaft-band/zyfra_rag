import json
import logging
import os
from contextlib import closing
from dataclasses import dataclass
from time import monotonic, sleep
from uuid import UUID, uuid4

import psycopg
from confluent_kafka import Consumer, KafkaError, KafkaException, Producer
from confluent_kafka.admin import AdminClient, NewTopic
from redis import Redis
from redis.exceptions import RedisError, ResponseError

from ..generation.service import AskService
from ..observability import observation
from ..web.answers import serialize_answer
from .storage import LEASE_SECONDS, JobEvent, JobStore


logger = logging.getLogger(__name__)
GROUP = "rag-workers"
DISPATCH_GROUP = "rag-dispatch"
BROKER_ERRORS = (KafkaException, RedisError, psycopg.Error, TimeoutError)


@dataclass(frozen=True)
class QueueSettings:
    kafka: str
    redis: str
    name: str

    @classmethod
    def from_environment(cls) -> QueueSettings:
        return cls(
            os.getenv("RAG_KAFKA_BOOTSTRAP_SERVERS", "127.0.0.1:9092"),
            os.getenv("RAG_REDIS_URL", "redis://127.0.0.1:6379/0"),
            os.getenv("RAG_QUEUE_NAME", "rag.jobs.v1"),
        )

    def redis_client(self) -> Redis:
        return Redis.from_url(self.redis, decode_responses=True, socket_timeout=5, socket_connect_timeout=5)


def ensure_group(redis: Redis, stream: str) -> None:
    try:
        redis.xgroup_create(stream, GROUP, id="0", mkstream=True)
    except ResponseError as error:
        if not str(error).startswith("BUSYGROUP"):
            raise


def initialize_queue(settings: QueueSettings) -> None:
    admin = AdminClient({"bootstrap.servers": settings.kafka})
    future = admin.create_topics([NewTopic(settings.name, num_partitions=3, replication_factor=1)], request_timeout=10)
    try:
        future[settings.name].result(timeout=12)
    except KafkaException as error:
        if error.args[0].code() != KafkaError.TOPIC_ALREADY_EXISTS:
            raise
    with closing(settings.redis_client()) as redis:
        ensure_group(redis, settings.name)


class KafkaPublisher:
    def __init__(self, settings: QueueSettings) -> None:
        self._topic = settings.name
        self._producer = Producer({
            "bootstrap.servers": settings.kafka,
            "enable.idempotence": True,
            "delivery.timeout.ms": 5000,
        })

    def send(self, event: JobEvent) -> None:
        delivery = []
        self._producer.produce(
            self._topic,
            key=str(event.conversation_id),
            value=json.dumps({
                "event_id": event.event_id, "kind": event.kind,
                "job_id": str(event.job_id), "conversation_id": str(event.conversation_id),
            }),
            on_delivery=lambda error, _: delivery.append(error),
        )
        if self._producer.flush(6) or not delivery:
            raise TimeoutError("Kafka delivery timed out")
        if delivery[0] is not None:
            raise KafkaException(delivery[0])


def run_publisher(store: JobStore, settings: QueueSettings) -> None:
    publisher = KafkaPublisher(settings)
    recover_at = 0.0
    while True:
        try:
            if monotonic() >= recover_at:
                store.recover()
                recover_at = monotonic() + 5
            if not store.publish_one(publisher.send):
                sleep(0.25)
        except BROKER_ERRORS as error:
            logger.warning("Queue publisher retry (%s)", type(error).__name__)
            sleep(2)


class KafkaDispatcher:
    def __init__(self, settings: QueueSettings) -> None:
        self._stream = settings.name
        self._consumer = Consumer({
            "bootstrap.servers": settings.kafka,
            "group.id": f"{settings.name}.{DISPATCH_GROUP}",
            "enable.auto.commit": False,
            "auto.offset.reset": "earliest",
        })
        self._consumer.subscribe([settings.name])

    def close(self) -> None:
        self._consumer.close()

    def step(self, redis: Redis, *, timeout: float = 1) -> bool:
        record = self._consumer.poll(timeout)
        if record is None:
            return False
        if record.error() is not None:
            raise KafkaException(record.error())
        try:
            event = json.loads(record.value())
            job_id = str(UUID(event["job_id"])) if event["kind"] == "job.ready" else None
        except (KeyError, ValueError, TypeError, AttributeError):
            logger.warning("Invalid Kafka job event at partition=%s offset=%s", record.partition(), record.offset())
            job_id = None
        if job_id is not None:
            redis.xadd(self._stream, {"job_id": job_id})
        # On failure, leave this offset uncommitted and reopen the consumer from its committed position.
        offsets = self._consumer.commit(message=record, asynchronous=False)
        for offset in offsets or ():
            if offset.error is not None:
                raise KafkaException(offset.error)
        return True


def run_dispatcher(settings: QueueSettings) -> None:
    while True:
        try:
            with closing(settings.redis_client()) as redis, closing(KafkaDispatcher(settings)) as dispatcher:
                while True:
                    dispatcher.step(redis)
        except BROKER_ERRORS as error:
            logger.warning("Queue dispatcher retry (%s)", type(error).__name__)
            sleep(2)


class RedisWorker:
    def __init__(self, redis: Redis, stream: str, store: JobStore, service: AskService) -> None:
        self._redis = redis
        self._stream = stream
        self._store = store
        self._service = service
        self._consumer = uuid4().hex
        self._reclaim_cursor = "0-0"
        self._reclaim_at = 0.0
        ensure_group(redis, stream)

    def step(self, *, block_ms: int = 1000) -> bool:
        entries = []
        if monotonic() >= self._reclaim_at:
            self._reclaim_cursor, entries, *_ = self._redis.xautoclaim(
                self._stream, GROUP, self._consumer, LEASE_SECONDS * 1000, self._reclaim_cursor, count=1,
            )
            if self._reclaim_cursor == "0-0":
                self._reclaim_at = monotonic() + 5
        if not entries:
            streams = self._redis.xreadgroup(GROUP, self._consumer, {self._stream: ">"}, count=1, block=block_ms)
            entries = streams[0][1] if streams else []
        if not entries:
            return False
        entry_id, fields = entries[0]
        try:
            job_id = UUID(fields["job_id"])
        except (ValueError, KeyError):
            logger.warning("Invalid Redis job entry id=%s", entry_id)
        else:
            self._process(job_id)
        with self._redis.pipeline(transaction=True) as transaction:
            transaction.xack(self._stream, GROUP, entry_id)
            transaction.xdel(self._stream, entry_id)
            transaction.execute()
        return True

    def _process(self, job_id: UUID) -> None:
        claim = self._store.claim(job_id)
        if claim is None:
            return
        with observation("rag.job") as trace:
            try:
                result = serialize_answer(self._service.ask(claim.question, history=claim.context.history))
            except Exception as error:
                logger.warning("Job %s processing failed (%s)", job_id, type(error).__name__)
                self._store.retry(claim)
                trace(level="WARNING", metadata={"job_id": str(job_id), "outcome": "processing_error"})
            else:
                saved = self._store.complete(claim, result.history_text(), result.model_dump(mode="json"))
                trace(metadata={"job_id": str(job_id), "outcome": "saved" if saved else "stale_claim"})


def run_worker(store: JobStore, service: AskService, settings: QueueSettings) -> None:
    while True:
        try:
            with closing(settings.redis_client()) as redis:
                worker = RedisWorker(redis, settings.name, store, service)
                while True:
                    worker.step()
        except BROKER_ERRORS as error:
            logger.warning("Queue worker retry (%s)", type(error).__name__)
            sleep(2)
