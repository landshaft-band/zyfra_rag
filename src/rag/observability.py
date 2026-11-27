import json
import logging
import math
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import cache


logger = logging.getLogger(__name__)
_MODEL_UPDATE = ContextVar("model_trace_update", default=None)
_DEFERRED_OUTPUT = ContextVar("deferred_model_output", default=None)
_CREDENTIAL = re.compile(
    r"""(?i)(\bbearer\s+|\b(?:api[_-]?key|access[_-]?token|secret[_-]?key|password)["']?\s*[:=]\s*["']?)[^\s"'<>;,\\]+"""
)
_TOKEN = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})\b")


def trace_content_enabled() -> bool:
    return os.getenv("RAG_TRACE_CONTENT", "false").strip().lower() == "true"


def _redact(value, known_secrets: tuple[str, ...]):
    if isinstance(value, str):
        for secret in known_secrets:
            value = value.replace(secret, "[REDACTED]")
        return _TOKEN.sub("[REDACTED]", _CREDENTIAL.sub(r"\1[REDACTED]", value))
    if isinstance(value, dict):
        return {key: _redact(text, known_secrets) for key, text in value.items()}
    if isinstance(value, (tuple, list)):
        return [_redact(text, known_secrets) for text in value]
    return value


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

    metadata = {}

    def update(**fields):
        if active is None:
            return
        if not trace_content_enabled():
            fields.pop("input", None)
            fields.pop("output", None)
        elif "input" in fields or "output" in fields:
            secrets = tuple(
                secret for name in ("RAG_LLM_API_KEY", "RAG_RERANK_API_KEY", "LANGFUSE_SECRET_KEY")
                if (secret := os.getenv(name, "").strip())
            )
            for name in ("input", "output"):
                if name in fields:
                    fields[name] = _redact(fields[name], secrets)
        if "metadata" in fields:
            metadata.update(fields["metadata"])
            fields["metadata"] = dict(metadata)
        if fields:
            _try_telemetry(active.update, **fields)

    token = _MODEL_UPDATE.set(update) if kind == "generation" else None
    try:
        yield update
    finally:
        if token is not None:
            _MODEL_UPDATE.reset(token)
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


def _valid_amount(value) -> bool:
    return type(value) in (int, float) and 0 <= value < math.inf


def cost_fields(model: str, usage: dict) -> dict:
    metadata = {}
    fields = {"metadata": metadata}
    for currency in ("rub", "usd"):
        if _valid_amount(amount := usage.get(f"cost_{currency}")):
            metadata[f"cost_{currency}"] = amount
            if currency == "usd":
                fields["cost_details"] = {"total": amount}
    if metadata:
        metadata["cost_source"] = "provider"
        return fields

    try:
        catalog = json.loads(os.getenv("RAG_MODEL_PRICES_JSON", "{}"))
        tariff = catalog.get(model, {}) if isinstance(catalog, dict) else {}
    except ValueError:
        tariff = {}
    if not isinstance(tariff, dict):
        return {}
    currency = str(tariff.get("currency", "")).lower()
    input_rate = tariff.get("input_per_million")
    output_rate = tariff.get("output_per_million")
    counts = usage_details(usage)
    if (
        currency not in ("rub", "usd")
        or not _valid_amount(input_rate) or not _valid_amount(output_rate)
        or "input" not in counts or "output" not in counts
    ):
        return {}
    amount = (counts["input"] * input_rate + counts["output"] * output_rate) / 1_000_000
    return {"metadata": {
        "cost_source": "estimate", f"estimated_cost_{currency}": amount, "tariff_currency": currency,
        "input_per_million": input_rate, "output_per_million": output_rate,
    }}


def record_model_input(messages: list[dict], api_key: str | None) -> None:
    if (update := _MODEL_UPDATE.get()) is not None and trace_content_enabled():
        update(input=_redact(messages, (api_key,)) if api_key else messages)


def record_model_result(model: str, usage: dict | None, content: str | None, api_key: str | None) -> None:
    update = _MODEL_UPDATE.get()
    if update is None:
        return
    if usage is not None:
        update(usage_details=usage_details(usage), **cost_fields(model, usage))
    if not isinstance(content, str) or not trace_content_enabled():
        return
    safe_content = _redact(content, (api_key,)) if api_key else content
    if (pending := _DEFERRED_OUTPUT.get()) is not None:
        pending.append(safe_content)
    else:
        update(output=safe_content)


@contextmanager
def defer_model_output() -> Iterator[list[str]]:
    pending = []
    token = _DEFERRED_OUTPUT.set(pending)
    try:
        yield pending
    finally:
        _DEFERRED_OUTPUT.reset(token)
