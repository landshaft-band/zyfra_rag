import json
import re
from collections.abc import Collection
from dataclasses import dataclass
from typing import Literal


INLINE_CITATION = re.compile(r"\[\s*S[0-9]+\b", re.IGNORECASE)


@dataclass(frozen=True)
class AnswerItem:
    text: str
    source_ids: tuple[str, ...]


@dataclass(frozen=True)
class Answer:
    status: Literal["answer", "clarify", "no_answer"]
    items: tuple[AnswerItem, ...]
    missing_information: tuple[str, ...]


def parse_answer(content: str, allowed_sources: Collection[str]) -> Answer:
    """Validate the model's JSON and the references it is allowed to cite."""
    try:
        record = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError("Ответ модели не удалось прочитать как JSON.") from error
    if not isinstance(record, dict):
        raise ValueError("Ожидался JSON-объект ответа.")
    status = record.get("status")
    if status not in ("answer", "clarify", "no_answer"):
        raise ValueError("Неизвестный статус ответа модели.")

    items = record.get("answer_items")
    if not isinstance(items, list):
        raise ValueError("Ожидался список answer_items.")
    missing = record.get("missing_information", [])
    if isinstance(missing, str):
        missing = [missing]
    details = (
        tuple(value.strip() for value in missing if isinstance(value, str) and value.strip())
        if isinstance(missing, list) else ()
    )
    if status == "answer" and not items:
        raise ValueError("Для статуса answer нужен хотя бы один пункт ответа.")
    if status != "answer" and items:
        raise ValueError("Пункты ответа разрешены только для статуса answer.")
    if status == "clarify" and not details:
        raise ValueError("Для статуса clarify укажите, что нужно уточнить.")

    parsed = tuple(_parse_item(item, allowed_sources) for item in items)
    return Answer(status, parsed, details)


def _parse_item(item: object, allowed_sources: Collection[str]) -> AnswerItem:
    if not isinstance(item, dict):
        raise ValueError("Ожидался JSON-объект пункта ответа.")
    text = item.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("У пункта ответа должен быть непустой текст.")
    if INLINE_CITATION.search(text):
        raise ValueError("Метки источников должны находиться в source_ids.")
    references = item.get("source_ids")
    if not isinstance(references, list) or not references:
        raise ValueError("Каждый пункт ответа должен ссылаться на источник.")
    if not all(isinstance(ref, str) and ref in allowed_sources for ref in references):
        raise ValueError("Ответ ссылается на источник вне переданного контекста.")
    return AnswerItem(text.strip(), tuple(dict.fromkeys(references)))
