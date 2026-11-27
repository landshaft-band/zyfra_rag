from pathlib import PureWindowsPath
from typing import Annotated

from fastapi import HTTPException
from pydantic import BaseModel, StringConstraints
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from ..generation.answer import AnswerItem
from ..generation.service import AskResult, AskService, AskStatus


SEARCH_ERRORS = (OSError, ResponseHandlingException, UnexpectedResponse)


class AskRequest(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class SourceView(BaseModel):
    source_id: str
    title: str
    text: str
    file_name: str
    locator: tuple[str, ...]
    start_line: int | None


class AskResponse(BaseModel):
    status: AskStatus
    items: tuple[AnswerItem, ...]
    sources: list[SourceView]
    missing_information: tuple[str, ...]
    message: str | None

    def history_text(self) -> str:
        return "\n".join([item.text for item in self.items] + list(self.missing_information))


def ask_question(service: AskService, question: str, *, history: tuple[tuple[str, str], ...] = ()) -> AskResponse:
    try:
        result = service.ask(question, history=history)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    except SEARCH_ERRORS as error:
        raise HTTPException(503, "Поиск по базе знаний сейчас недоступен.") from error

    return serialize_answer(result)


def serialize_answer(result: AskResult) -> AskResponse:
    return AskResponse(
        status=result.status,
        items=result.items,
        sources=[
            SourceView(
                source_id=source.source_id,
                title=source.title,
                text=source.text,
                file_name=PureWindowsPath(source.file_path).name,
                locator=source.locator,
                start_line=source.start_line,
            )
            for source in result.sources
        ],
        missing_information=result.missing_information,
        message=result.message,
    )
