import re
from unicodedata import normalize

from guardrails import OnFailAction, Validator, register_validator
from guardrails.settings import settings
from guardrails_ai.types import FailResult, PassResult

from .generation.client import SYSTEM_PROMPT


def _normalize(text: str) -> str:
    return " ".join(normalize("NFKC", text).casefold().split())


INPUT_RULES = (
    ("instruction_override", re.compile(
        r"\b(?:игнорируй(?:те)?|проигнорируй(?:те)?|забудь(?:те)?|отмени(?:те)?|"
        r"ignore|disregard|override)\b.{0,70}?"
        r"\b(?:инструкц\w*|правил\w*|ограничени\w*|instructions?|rules?|restrictions?)\b"
    )),
    ("system_prompt_request", re.compile(
        r"\b(?:покажи(?:те)?|раскрой(?:те)?|выведи(?:те)?|напечатай(?:те)?|"
        r"show|reveal|print|disclose)\b.{0,70}?"
        r"\b(?:системн\w*\s+промпт\w*|скрыт\w*\s+инструкц\w*|"
        r"system\s+prompt|hidden\s+instructions?)\b"
    )),
)
PROMPT_MARKERS = tuple(
    _normalize(line) for line in SYSTEM_PROMPT.splitlines()
    if len(line.strip()) >= 55
)

# The application exports only its own metadata through Langfuse.
settings.rc.enable_metrics = False
settings.disable_tracing = True


@register_validator(name="rag/prompt-policy", data_type="string")
class PromptPolicy(Validator):
    def _validate(self, value: str, metadata: dict) -> PassResult | FailResult:
        text = _normalize(value)
        if metadata["stage"] == "input":
            for reason, pattern in INPUT_RULES:
                if pattern.search(text):
                    return FailResult(error_message=reason)
        elif sum(marker in text for marker in PROMPT_MARKERS) >= 2:
            return FailResult(error_message="system_prompt_excerpt")
        return PassResult()


prompt_policy = PromptPolicy(on_fail=OnFailAction.NOOP, use_local=True)


def policy_reason(stage: str, text: str) -> str | None:
    result = prompt_policy.validate(text, {"stage": stage})
    return result.error_message if isinstance(result, FailResult) else None
