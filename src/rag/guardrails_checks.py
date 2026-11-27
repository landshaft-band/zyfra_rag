import logging
import os
from time import perf_counter
from typing import Literal

from .observability import observation


logger = logging.getLogger(__name__)
UNAVAILABLE_MESSAGE = "Проверка безопасности временно недоступна."


def guardrail_mode() -> Literal["off", "observe", "enforce"]:
    mode = os.getenv("RAG_GUARDRAILS_MODE", "observe").strip().lower()
    return mode if mode in ("off", "observe", "enforce") else "observe"


def check_guardrail(stage: Literal["input", "output"], text: str) -> str | None:
    mode = guardrail_mode()
    if not text or mode == "off":
        return None
    enforce = mode == "enforce"

    with observation(f"guardrails.{stage}") as trace:
        if stage == "input":
            trace(input=text)
        started = perf_counter()
        try:
            os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
            from .safety_validators import policy_reason

            reason = policy_reason(stage, text)
        except Exception as error:
            elapsed = round((perf_counter() - started) * 1000, 2)
            trace(level="WARNING", metadata={
                "decision": "error", "reason": type(error).__name__, "elapsed_ms": elapsed,
                "blocked": enforce,
            })
            logger.warning("Guardrails %s validation unavailable (%s)", stage, type(error).__name__)
            return UNAVAILABLE_MESSAGE if enforce else None

        elapsed = round((perf_counter() - started) * 1000, 2)
        if reason is not None:
            trace(level="WARNING", metadata={
                "decision": "flag", "reason": reason, "elapsed_ms": elapsed, "blocked": enforce,
            })
            logger.warning("Guardrails %s %s %s (%.2f ms)", stage, "blocked" if enforce else "flagged", reason, elapsed)
            if enforce:
                if stage == "input":
                    return "Запрос остановлен проверкой безопасности. Задайте вопрос по работе продукта."
                return "Ответ остановлен проверкой безопасности."
        else:
            trace(
                metadata={"decision": "pass", "elapsed_ms": elapsed},
                **({"input": text} if stage == "output" else {}),
            )
    return None
