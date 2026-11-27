from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from importlib.resources import files
from typing import Literal
from uuid import UUID, uuid4

import psycopg
from psycopg.rows import class_row
from psycopg.types.json import Jsonb


@dataclass(frozen=True)
class Conversation:
    id: UUID
    created_at: datetime


@dataclass(frozen=True)
class Message:
    position: int
    role: Literal["user", "assistant"]
    content: str
    result: dict | None
    created_at: datetime


@dataclass(frozen=True)
class TurnContext:
    next_position: int
    history: tuple[tuple[str, str], ...]


def postgres_dsn() -> str | None:
    return os.getenv("RAG_POSTGRES_DSN", "").strip() or None


def migrate(dsn: str) -> tuple[str, ...]:
    directory = files("rag.conversations").joinpath("migrations")
    scripts = sorted(
        (script for script in directory.iterdir() if script.name.endswith(".sql")),
        key=lambda script: script.name,
    )
    if not scripts:
        raise RuntimeError("Файлы миграций PostgreSQL отсутствуют.")
    applied = []
    with psycopg.connect(dsn, connect_timeout=5) as connection:
        connection.execute("SELECT pg_advisory_xact_lock(713809321)")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_schema_migrations (
                name text PRIMARY KEY,
                applied_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        known = {name for (name,) in connection.execute("SELECT name FROM rag_schema_migrations")}
        for script in scripts:
            if script.name in known:
                continue
            connection.execute(script.read_text(encoding="utf-8"))
            connection.execute("INSERT INTO rag_schema_migrations (name) VALUES (%s)", (script.name,))
            applied.append(script.name)
    return tuple(applied)


def read_turn_context(connection: psycopg.Connection, conversation_id: UUID, position: int) -> TurnContext:
    messages = connection.execute(
        """
        SELECT m.role, left(m.content, 6000) FROM messages AS m
        WHERE m.conversation_id = %s AND m.position < %s
          AND NOT EXISTS (
              SELECT 1 FROM messages AS answer
              WHERE answer.conversation_id = m.conversation_id
                AND answer.position = m.position + CASE WHEN m.role = 'user' THEN 1 ELSE 0 END
                AND answer.position < %s
                AND answer.role = 'assistant'
                AND answer.result->>'status' = 'blocked'
          )
        ORDER BY m.position DESC LIMIT 6
        """,
        (conversation_id, position, position),
    ).fetchall()
    return TurnContext(position, tuple(reversed(messages)))


def write_turn(
    connection: psycopg.Connection, conversation_id: UUID, position: int,
    question: str, answer: str, result: dict,
) -> None:
    """Append a complete turn with the conversation row already locked."""
    connection.execute(
        """
        INSERT INTO messages (conversation_id, position, role, content, result)
        VALUES (%s, %s, 'user', %s, NULL), (%s, %s, 'assistant', %s, %s)
        """,
        (conversation_id, position, question, conversation_id, position + 1, answer, Jsonb(result)),
    )
    connection.execute(
        "UPDATE conversations SET next_position = %s WHERE id = %s",
        (position + 2, conversation_id),
    )


class ConversationStore:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def create(self, owner_digest: str) -> Conversation:
        with (
            psycopg.connect(self._dsn, connect_timeout=5) as connection,
            connection.cursor(row_factory=class_row(Conversation)) as cursor,
        ):
            cursor.execute(
                "INSERT INTO conversations (id, owner_digest) VALUES (%s, %s) RETURNING id, created_at",
                (uuid4(), owner_digest),
            )
            return next(cursor)

    def list(self, owner_digest: str) -> list[Conversation]:
        with (
            psycopg.connect(self._dsn, connect_timeout=5) as connection,
            connection.cursor(row_factory=class_row(Conversation)) as cursor,
        ):
            cursor.execute(
                "SELECT id, created_at FROM conversations WHERE owner_digest = %s ORDER BY created_at DESC, id DESC",
                (owner_digest,),
            )
            return cursor.fetchall()

    def history(self, conversation_id: UUID, owner_digest: str) -> tuple[Conversation, list[Message]] | None:
        with psycopg.connect(self._dsn, connect_timeout=5) as connection:
            with connection.cursor(row_factory=class_row(Conversation)) as cursor:
                cursor.execute(
                    "SELECT id, created_at FROM conversations WHERE id = %s AND owner_digest = %s",
                    (conversation_id, owner_digest),
                )
                conversation = cursor.fetchone()
            if conversation is None:
                return None
            with connection.cursor(row_factory=class_row(Message)) as cursor:
                cursor.execute(
                    """
                    SELECT position, role, content, result, created_at
                    FROM messages WHERE conversation_id = %s ORDER BY position
                    """,
                    (conversation_id,),
                )
                return conversation, cursor.fetchall()

    def turn_context(self, conversation_id: UUID, owner_digest: str) -> TurnContext | None:
        with psycopg.connect(self._dsn, connect_timeout=5) as connection:
            row = connection.execute(
                "SELECT next_position FROM conversations WHERE id = %s AND owner_digest = %s",
                (conversation_id, owner_digest),
            ).fetchone()
            if row is None:
                return None
            return read_turn_context(connection, conversation_id, row[0])

    def has_pending_jobs(self, conversation_id: UUID, owner_digest: str) -> bool:
        with psycopg.connect(self._dsn, connect_timeout=5) as connection:
            return _has_pending_jobs(connection, conversation_id, owner_digest)

    def append_turn(
        self, conversation_id: UUID, owner_digest: str, expected_position: int,
        question: str, answer: str, result: dict,
    ) -> bool:
        with psycopg.connect(self._dsn, connect_timeout=5) as connection:
            claimed = connection.execute(
                """
                SELECT id FROM conversations
                WHERE id = %s AND owner_digest = %s AND next_position = %s FOR UPDATE
                """,
                (conversation_id, owner_digest, expected_position),
            ).fetchone()
            if claimed is None or _has_pending_jobs(connection, conversation_id, owner_digest):
                return False
            write_turn(connection, conversation_id, expected_position, question, answer, result)
            return True

    def append_message(
        self, conversation_id: UUID, owner_digest: str,
        role: Literal["user", "assistant"], content: str, result: dict | None = None,
    ) -> Message | None:
        with psycopg.connect(self._dsn, connect_timeout=5) as connection:
            position = connection.execute(
                """
                UPDATE conversations SET next_position = next_position + 1
                WHERE id = %s AND owner_digest = %s
                RETURNING next_position - 1
                """,
                (conversation_id, owner_digest),
            ).fetchone()
            if position is None:
                return None
            with connection.cursor(row_factory=class_row(Message)) as cursor:
                cursor.execute(
                    """
                    INSERT INTO messages (conversation_id, position, role, content, result)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING position, role, content, result, created_at
                    """,
                    (conversation_id, position[0], role, content, Jsonb(result) if result is not None else None),
                )
                return next(cursor)


def _has_pending_jobs(connection: psycopg.Connection, conversation_id: UUID, owner_digest: str) -> bool:
    return connection.execute(
        """
        SELECT 1 FROM jobs JOIN conversations ON conversations.id = jobs.conversation_id
        WHERE conversation_id = %s AND owner_digest = %s AND jobs.status IN ('queued', 'running')
        LIMIT 1
        """,
        (conversation_id, owner_digest),
    ).fetchone() is not None
