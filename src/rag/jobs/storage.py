from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from uuid import UUID, uuid4

import psycopg
from psycopg.rows import class_row

from ..conversations.storage import TurnContext, read_turn_context, write_turn


JobStatus = Literal["queued", "running", "completed", "failed"]
MAX_ATTEMPTS = 3
LEASE_SECONDS = 180
RESIGNAL_SECONDS = 60
FAILED_MESSAGE = "Не удалось обработать вопрос после нескольких попыток."


@dataclass(frozen=True)
class Job:
    id: UUID
    conversation_id: UUID
    request_id: UUID
    question: str
    status: JobStatus
    attempts: int
    result: dict | None
    error: str | None
    created_at: datetime


@dataclass(frozen=True)
class Claim:
    job_id: UUID
    conversation_id: UUID
    lease_id: UUID
    question: str
    context: TurnContext


@dataclass(frozen=True)
class JobEvent:
    event_id: int
    job_id: UUID
    conversation_id: UUID
    kind: str


class RequestConflict(ValueError):
    """A request ID was already used with a different question."""


JOB_VIEW = """
    SELECT j.id, j.conversation_id, j.request_id, j.question, j.status, j.attempts,
           m.result, j.error, j.created_at
    FROM jobs AS j JOIN conversations AS c ON c.id = j.conversation_id
    LEFT JOIN messages AS m
        ON m.conversation_id = j.conversation_id AND m.position = j.assistant_position
"""


class JobStore:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def submit(self, conversation_id: UUID, owner: str, request_id: UUID, question: str) -> Job | None:
        with psycopg.connect(self._dsn, connect_timeout=5) as db:
            if db.execute(
                "SELECT id FROM conversations WHERE id = %s AND owner_digest = %s FOR UPDATE",
                (conversation_id, owner),
            ).fetchone() is None:
                return None
            with db.cursor(row_factory=class_row(Job)) as cursor:
                cursor.execute(JOB_VIEW + " WHERE j.conversation_id = %s AND j.request_id = %s",
                               (conversation_id, request_id))
                existing = cursor.fetchone()
                if existing is not None:
                    if existing.question != question:
                        raise RequestConflict
                    return existing

                job_id = uuid4()
                is_head = _head(db, conversation_id) is None
                db.execute(
                    "INSERT INTO jobs (id, conversation_id, request_id, question) VALUES (%s, %s, %s, %s)",
                    (job_id, conversation_id, request_id, question),
                )
                if is_head:
                    _schedule(db, job_id)
                cursor.execute(JOB_VIEW + " WHERE j.id = %s", (job_id,))
                return next(cursor)

    def get(self, job_id: UUID, owner: str) -> Job | None:
        with psycopg.connect(self._dsn, connect_timeout=5) as db, db.cursor(row_factory=class_row(Job)) as cursor:
            cursor.execute(JOB_VIEW + " WHERE j.id = %s AND c.owner_digest = %s", (job_id, owner))
            return cursor.fetchone()

    def active(self, conversation_id: UUID, owner: str) -> list[Job] | None:
        with psycopg.connect(self._dsn, connect_timeout=5) as db, db.cursor(row_factory=class_row(Job)) as cursor:
            if db.execute(
                "SELECT 1 FROM conversations WHERE id = %s AND owner_digest = %s",
                (conversation_id, owner),
            ).fetchone() is None:
                return None
            cursor.execute(
                JOB_VIEW + " WHERE j.conversation_id = %s AND j.status IN ('queued', 'running') ORDER BY j.sequence",
                (conversation_id,),
            )
            return cursor.fetchall()

    def claim(self, job_id: UUID) -> Claim | None:
        with psycopg.connect(self._dsn, connect_timeout=5) as db:
            conversation = _lock_for_job(db, job_id)
            if conversation is None or _head(db, conversation[0]) != job_id:
                return None
            conversation_id, position = conversation
            lease_id = uuid4()
            row = db.execute(
                """
                UPDATE jobs SET status = 'running', attempts = attempts + 1, lease_id = %s,
                    leased_until = clock_timestamp() + %s * interval '1 second'
                WHERE id = %s AND status = 'queued' AND signal_at <= clock_timestamp()
                RETURNING question
                """,
                (lease_id, LEASE_SECONDS, job_id),
            ).fetchone()
            if row is None:
                return None
            return Claim(job_id, conversation_id, lease_id, row[0], read_turn_context(db, conversation_id, position))

    def complete(self, claim: Claim, answer: str, result: dict) -> bool:
        with psycopg.connect(self._dsn, connect_timeout=5) as db:
            conversation = _lock_for_job(db, claim.job_id)
            attempt = _live_attempt(db, claim) if conversation is not None else None
            if attempt is None:
                return False
            conversation_id, position = conversation
            if position != claim.context.next_position:
                _retry_or_fail(db, claim.job_id, conversation_id, attempt)
                return False
            write_turn(db, conversation_id, position, claim.question, answer, result)
            db.execute(
                """
                UPDATE jobs SET status = 'completed', assistant_position = %s,
                    lease_id = NULL, leased_until = NULL WHERE id = %s
                """,
                (position + 1, claim.job_id),
            )
            _terminal_event(db, claim.job_id, conversation_id, "job.completed")
            return True

    def retry(self, claim: Claim) -> None:
        with psycopg.connect(self._dsn, connect_timeout=5) as db:
            if _lock_for_job(db, claim.job_id) is not None and (attempt := _live_attempt(db, claim)) is not None:
                _retry_or_fail(db, claim.job_id, claim.conversation_id, attempt)

    def recover(self) -> int:
        with psycopg.connect(self._dsn, connect_timeout=5) as db:
            rows = db.execute(
                """
                SELECT c.id, j.id, j.status, j.attempts
                FROM jobs AS j JOIN conversations AS c ON c.id = j.conversation_id
                WHERE (
                    (j.status = 'running' AND j.leased_until <= clock_timestamp()) OR (
                        j.status = 'queued' AND j.signal_at <= clock_timestamp() - %s * interval '1 second'
                        AND NOT EXISTS (
                            SELECT 1 FROM job_outbox o WHERE o.job_id = j.id AND o.published_at IS NULL
                        )
                    )
                ) AND NOT EXISTS (
                    SELECT 1 FROM jobs AS earlier WHERE earlier.conversation_id = j.conversation_id
                      AND earlier.sequence < j.sequence AND earlier.status IN ('queued', 'running')
                )
                ORDER BY j.sequence LIMIT 25 FOR UPDATE OF c, j SKIP LOCKED
                """,
                (RESIGNAL_SECONDS,),
            ).fetchall()
            for conversation_id, job_id, status, attempts in rows:
                if status == "running":
                    _retry_or_fail(db, job_id, conversation_id, attempts)
                else:
                    _schedule(db, job_id)
            return len(rows)

    def publish_one(self, deliver: Callable[[JobEvent], None]) -> bool:
        with (
            psycopg.connect(self._dsn, connect_timeout=5) as db,
            db.cursor(row_factory=class_row(JobEvent)) as cursor,
        ):
            cursor.execute(
                """
                SELECT o.id AS event_id, o.job_id, j.conversation_id, o.kind
                FROM job_outbox AS o JOIN jobs AS j ON j.id = o.job_id
                WHERE o.published_at IS NULL AND o.available_at <= clock_timestamp()
                ORDER BY o.id LIMIT 1 FOR UPDATE OF o SKIP LOCKED
                """,
            )
            event = cursor.fetchone()
            if event is None:
                return False
            deliver(event)
            db.execute("UPDATE job_outbox SET published_at = clock_timestamp() WHERE id = %s", (event.event_id,))
            return True


def _lock_for_job(db: psycopg.Connection, job_id: UUID):
    return db.execute(
        """
        SELECT c.id, c.next_position FROM jobs AS j JOIN conversations AS c ON c.id = j.conversation_id
        WHERE j.id = %s FOR UPDATE OF c
        """,
        (job_id,),
    ).fetchone()


def _head(db: psycopg.Connection, conversation_id: UUID) -> UUID | None:
    row = db.execute(
        """
        SELECT id FROM jobs WHERE conversation_id = %s AND status IN ('queued', 'running')
        ORDER BY sequence LIMIT 1
        """,
        (conversation_id,),
    ).fetchone()
    return row[0] if row else None


def _live_attempt(db: psycopg.Connection, claim: Claim) -> int | None:
    row = db.execute(
        "SELECT attempts FROM jobs WHERE id = %s AND lease_id = %s AND leased_until > clock_timestamp()",
        (claim.job_id, claim.lease_id),
    ).fetchone()
    return row[0] if row else None


def _schedule(db: psycopg.Connection, job_id: UUID, delay: int = 0) -> None:
    due = db.execute(
        "UPDATE jobs SET signal_at = clock_timestamp() + %s * interval '1 second' WHERE id = %s RETURNING signal_at",
        (delay, job_id),
    ).fetchone()[0]
    db.execute("INSERT INTO job_outbox (job_id, kind, available_at) VALUES (%s, 'job.ready', %s)", (job_id, due))


def _terminal_event(db: psycopg.Connection, job_id: UUID, conversation_id: UUID, kind: str) -> None:
    db.execute("INSERT INTO job_outbox (job_id, kind) VALUES (%s, %s)", (job_id, kind))
    next_job = _head(db, conversation_id)
    if next_job is not None:
        _schedule(db, next_job)


def _retry_or_fail(db: psycopg.Connection, job_id: UUID, conversation_id: UUID, attempt: int) -> None:
    if attempt < MAX_ATTEMPTS:
        db.execute(
            "UPDATE jobs SET status = 'queued', lease_id = NULL, leased_until = NULL WHERE id = %s", (job_id,),
        )
        _schedule(db, job_id, delay=2 ** attempt)
    else:
        db.execute(
            "UPDATE jobs SET status = 'failed', error = %s, lease_id = NULL, leased_until = NULL WHERE id = %s",
            (FAILED_MESSAGE, job_id),
        )
        _terminal_event(db, job_id, conversation_id, "job.failed")
