import hashlib
import re
import secrets
from dataclasses import dataclass
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from ..conversations.storage import Conversation, ConversationStore, Message
from .answers import AskRequest, AskResponse, ask_question


router = APIRouter(prefix="/api/conversations", tags=["conversations"])
OWNER_COOKIE = "rag_conversation_owner"
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_-]{43}")


@dataclass(frozen=True)
class ConversationHistory:
    conversation: Conversation
    messages: list[Message]


def get_store(request: Request) -> ConversationStore:
    store = request.app.state.conversations
    if store is None:
        raise HTTPException(503, "Хранилище диалогов не настроено.", headers={"X-Rag-History": "disabled"})
    return store


Store = Annotated[ConversationStore, Depends(get_store)]


def owner_digest(token: str) -> str:
    return hashlib.sha256(token.encode("ascii")).hexdigest()


def owner_token(request: Request) -> str | None:
    token = request.cookies.get(OWNER_COOKIE, "")
    return token if TOKEN_PATTERN.fullmatch(token) else None


@router.post("", status_code=201)
def create_conversation(request: Request, response: Response, store: Store) -> Conversation:
    existing_token = owner_token(request)
    token = existing_token or secrets.token_urlsafe(32)
    conversation = store.create(owner_digest(token))
    if existing_token is None:
        response.set_cookie(
            OWNER_COOKIE, token, path="/api", max_age=30 * 24 * 60 * 60,
            httponly=True, samesite="strict", secure=request.url.scheme == "https",
        )
    response.headers["Cache-Control"] = "no-store"
    return conversation


@router.get("")
def list_conversations(request: Request, response: Response, store: Store) -> list[Conversation]:
    response.headers["Cache-Control"] = "no-store"
    token = owner_token(request)
    return store.list(owner_digest(token)) if token else []


@router.get("/{conversation_id}")
def get_history(conversation_id: UUID, request: Request, response: Response, store: Store) -> ConversationHistory:
    token = owner_token(request)
    history = store.history(conversation_id, owner_digest(token)) if token else None
    if history is None:
        raise HTTPException(404, "Диалог не найден.")
    response.headers["Cache-Control"] = "no-store"
    conversation, messages = history
    return ConversationHistory(conversation, messages)


@router.post("/{conversation_id}/turns")
def submit_turn(
    conversation_id: UUID, body: AskRequest, request: Request, response: Response, store: Store,
) -> AskResponse:
    token = owner_token(request)
    owner = owner_digest(token) if token else None
    context = store.turn_context(conversation_id, owner) if owner else None
    if context is None:
        raise HTTPException(404, "Диалог не найден.")
    if store.has_pending_jobs(conversation_id, owner):
        raise HTTPException(409, "В диалоге уже есть задания в очереди.")

    result = ask_question(request.app.state.ask_service, body.question, history=context.history)
    if not store.append_turn(
        conversation_id, owner, context.next_position,
        body.question, result.history_text(), result.model_dump(mode="json"),
    ):
        raise HTTPException(409, "Диалог уже обновился в другом запросе. Проверьте историю и отправьте вопрос снова.")
    response.headers["Cache-Control"] = "no-store"
    return result
