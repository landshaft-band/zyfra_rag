import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache


logger = logging.getLogger(__name__)


def _try_telemetry(operation, *args, **kwargs):
    try:
        return operation(*args, **kwargs)
    except Exception as error:
        logger.warning("Langfuse operation failed (%s)", type(error).__name__)
        return None


@cache
def _client():
    if os.getenv("LANGFUSE_TRACING_ENABLED", "true").strip().lower() == "false":
        return None
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return None

    def create():
        from langfuse import Langfuse

        return Langfuse()

    return _try_telemetry(create)


@contextmanager
def observation(name: str, kind: str = "span") -> Iterator:
    client = _client()
    manager = None
    active = None
    if client is not None:
        manager = _try_telemetry(client.start_as_current_observation, name=name, as_type=kind)
        if manager is not None:
            active = _try_telemetry(manager.__enter__)

    def update(**fields):
        if active is not None:
            _try_telemetry(active.update, **fields)

    try:
        yield update
    finally:
        if active is not None:
            # Do not export raw exception text, which may contain request details.
            _try_telemetry(manager.__exit__, None, None, None)


def flush() -> None:
    if _client.cache_info().currsize and (client := _client()) is not None:
        _try_telemetry(client.flush)


def usage_details(usage: dict) -> dict[str, int]:
    fields = {"prompt_tokens": "input", "completion_tokens": "output", "total_tokens": "total"}
    return {
        target: count
        for source, target in fields.items()
        if type(count := usage.get(source)) is int and count >= 0
    }
