from contextlib import asynccontextmanager
from pathlib import Path

import psycopg
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from qdrant_client import QdrantClient

from ..conversations.storage import ConversationStore, postgres_dsn
from ..generation.service import AskService
from ..jobs.storage import JobStore
from ..observability import flush
from ..retrieval.hybrid import HybridRetriever
from .answers import SEARCH_ERRORS, AskRequest, AskResponse, ask_question
from .conversations import router as conversations_router
from .jobs import router as jobs_router


STATIC_DIRECTORY = Path(__file__).with_name("static")


def create_app(qdrant_url: str, stats_directory: Path) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        client = QdrantClient(url=qdrant_url)
        try:
            retriever = HybridRetriever(client, stats_directory)
            app.state.client = client
            app.state.collection = retriever.collection
            app.state.ask_service = AskService(retriever)
            yield
        finally:
            client.close()
            flush()

    app = FastAPI(title="Помощник техподдержки", lifespan=lifespan)
    dsn = postgres_dsn()
    app.state.conversations = ConversationStore(dsn) if dsn else None
    app.state.jobs = JobStore(dsn) if dsn else None
    app.include_router(conversations_router)
    app.include_router(jobs_router)
    app.mount("/static", StaticFiles(directory=STATIC_DIRECTORY), name="static")

    @app.exception_handler(psycopg.Error)
    async def postgres_error(request: Request, error: psycopg.Error) -> JSONResponse:
        return JSONResponse(
            status_code=503, content={"detail": "Хранилище диалогов сейчас недоступно."},
        )

    @app.get("/", include_in_schema=False)
    def home() -> FileResponse:
        return FileResponse(STATIC_DIRECTORY / "index.html")

    @app.get("/api/health")
    def health(request: Request) -> dict[str, str]:
        try:
            ready = request.app.state.client.collection_exists(request.app.state.collection)
        except SEARCH_ERRORS:
            ready = False
        if not ready:
            raise HTTPException(503, "База знаний недоступна.")
        return {"status": "ready"}

    @app.post("/api/ask")
    def ask(body: AskRequest, request: Request) -> AskResponse:
        return ask_question(request.app.state.ask_service, body.question)

    return app
