from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, StringConstraints

from ..jobs.storage import Job, JobStore, RequestConflict
from .conversations import owner_digest, owner_token


router = APIRouter(prefix="/api", tags=["jobs"])


class JobRequest(BaseModel):
    request_id: UUID
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]


def get_jobs(request: Request) -> JobStore:
    store = request.app.state.jobs
    if store is None:
        raise HTTPException(503, "Хранилище заданий не настроено.")
    return store


Jobs = Annotated[JobStore, Depends(get_jobs)]


def get_owner(request: Request) -> str:
    token = owner_token(request)
    if token is None:
        raise HTTPException(404, "Диалог или задание не найдено.")
    return owner_digest(token)


Owner = Annotated[str, Depends(get_owner)]


@router.post("/conversations/{conversation_id}/jobs", status_code=202)
def submit(conversation_id: UUID, body: JobRequest, response: Response, store: Jobs, owner: Owner) -> Job:
    try:
        job = store.submit(conversation_id, owner, body.request_id, body.question)
    except RequestConflict as error:
        raise HTTPException(409, "Этот ключ запроса уже использован для другого вопроса.") from error
    if job is None:
        raise HTTPException(404, "Диалог не найден.")
    response.headers["Cache-Control"] = "no-store"
    response.headers["Location"] = f"/api/jobs/{job.id}"
    return job


@router.get("/conversations/{conversation_id}/jobs")
def list_active(conversation_id: UUID, response: Response, store: Jobs, owner: Owner) -> list[Job]:
    jobs = store.active(conversation_id, owner)
    if jobs is None:
        raise HTTPException(404, "Диалог не найден.")
    response.headers["Cache-Control"] = "no-store"
    return jobs


@router.get("/jobs/{job_id}")
def status(job_id: UUID, response: Response, store: Jobs, owner: Owner) -> Job:
    job = store.get(job_id, owner)
    if job is None:
        raise HTTPException(404, "Задание не найдено.")
    response.headers["Cache-Control"] = "no-store"
    return job
