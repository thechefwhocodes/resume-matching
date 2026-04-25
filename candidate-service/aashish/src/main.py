"""FastAPI entrypoint.

Lifespan responsibilities:
  1. Connect to Postgres
  2. CREATE EXTENSION IF NOT EXISTS vector
  3. Base.metadata.create_all (no Alembic by design - documented in EVALUATION.md)
  4. ingest_jobs_if_empty -- pulls data/jobs.json, strips HTML, embeds (or zero-vectors)
  5. (P1a+) build BM25 index in memory and stash on app.state
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from src.config import get_settings
from src.db.engine import SessionLocal, get_engine
from src.db.models import Base
from src.db.queries import count_jobs
from src.ingest import ingest_jobs_if_empty
from src.llm.client import get_llm_client
from src.pipeline import run_pipeline
from src.retrieval.bm25 import BM25Index
from src.schemas import ErrorResponse, MatchRequest, MatchResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
log = logging.getLogger("aashish.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    log.info("startup: database_url=%s", settings.database_url)
    log.info("startup: openai_key_present=%s", settings.has_openai_key)
    log.info("startup: default_approach=%s", settings.default_approach())

    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(bind=engine)

    llm = get_llm_client()
    with SessionLocal() as session:
        inserted = ingest_jobs_if_empty(session, llm, jobs_path=settings.jobs_json_path)
        total = count_jobs(session)
        log.info("startup: ingest done (inserted=%d, total_jobs=%d)", inserted, total)
        bm25_index = BM25Index.build_from_session(session)

    app.state.llm = llm
    app.state.bm25_index = bm25_index
    yield
    log.info("shutdown")


app = FastAPI(
    title="Resume Matching Service (aashish)",
    description="Matches resumes to jobs using BM25 / embeddings / LLM rerank.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, object]:
    settings = get_settings()
    with SessionLocal() as session:
        n = count_jobs(session)
    return {
        "status": "ok",
        "jobs_in_db": n,
        "default_approach": settings.default_approach(),
        "openai_key_present": settings.has_openai_key,
    }


@app.post("/match", response_model=MatchResponse)
def match(request: MatchRequest) -> MatchResponse:
    if not request.resume.content or not request.resume.content.strip():
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error="Bad Request",
                message="Resume content is required",
            ).model_dump(),
        )
    try:
        with SessionLocal() as session:
            return run_pipeline(
                request,
                session=session,
                llm=app.state.llm,
                bm25_index=app.state.bm25_index,
            )
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("match: pipeline failure")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="Internal Server Error",
                message="Matching pipeline failed",
                details=str(exc),
            ).model_dump(),
        ) from exc
