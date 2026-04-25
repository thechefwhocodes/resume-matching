"""Lazy seed: read data/jobs.json into Postgres on first boot.

Idempotent: if `jobs` already has rows, this is a no-op. We deliberately do
NOT use Alembic for the take-home (call it out as production work in
EVALUATION.md). Schema lives in `models.py` and is created by `Base.metadata.
create_all` in `main.py` lifespan before this function runs.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from src.db.models import Job
from src.db.queries import count_jobs
from src.llm.client import LLMClient
from src.utils.html import strip_html
from src.utils.salary import format_salary_range

log = logging.getLogger(__name__)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _embedding_text(record: dict[str, Any]) -> str:
    """Build the canonical text we embed (and BM25-index) for each job."""
    parts = [
        record.get("title") or "",
        record.get("job_category") or "",
        strip_html(record.get("responsibilities") or ""),
        record.get("requirements") or "",
    ]
    return "\n".join(p for p in parts if p)


def _row_from_record(record: dict[str, Any], embedding: list[float] | None) -> Job:
    responsibilities_html = record.get("responsibilities") or ""
    responsibilities_text = strip_html(responsibilities_html)
    salary_min = record.get("salary_min")
    salary_max = record.get("salary_max")
    salary_currency = record.get("salary_currency") or "$"
    return Job(
        job_id=record["job_id"],
        title=record.get("title") or "",
        company_name=record.get("company_name") or "",
        company_id=record.get("company_id"),
        responsibilities_html=responsibilities_html,
        responsibilities_text=responsibilities_text,
        requirements=record.get("requirements"),
        job_category=record.get("job_category"),
        location=record.get("location"),
        work_location_type=record.get("work_location_type"),
        yoe_min=record.get("yoe_min"),
        yoe_label=record.get("YOE"),
        salary_min=salary_min,
        salary_max=salary_max,
        salary_currency=salary_currency,
        salary_range=format_salary_range(salary_min, salary_max, salary_currency),
        equity_min=record.get("equity_min"),
        equity_max=record.get("equity_max"),
        employment_type=record.get("employment_type"),
        benefits=record.get("benefits"),
        h1b_sponsorship=bool(record.get("h1b_sponsorship", False)),
        status=record.get("status") or "active",
        created_at=record.get("created_at"),
        green_flags=record.get("greenFlags"),
        red_flags=record.get("redFlags"),
        ideal_companies=record.get("idealCompanies"),
        content_sha256=_sha256(_embedding_text(record)),
        embedding=embedding,
    )


def ingest_jobs_if_empty(
    session: Session,
    llm: LLMClient,
    *,
    jobs_path: str | Path,
) -> int:
    """Seed the jobs table from `jobs_path` if empty. Returns rows inserted."""
    existing = count_jobs(session)
    if existing > 0:
        log.info("ingest: skipped (%d jobs already in DB)", existing)
        return 0

    path = Path(jobs_path)
    if not path.exists():
        raise FileNotFoundError(f"jobs.json not found at {path.resolve()}")

    with path.open("r", encoding="utf-8") as f:
        records: list[dict[str, Any]] = json.load(f)

    log.info("ingest: loaded %d job records from %s", len(records), path)

    texts = [_embedding_text(r) for r in records]

    if llm.is_available:
        log.info(
            "ingest: computing real embeddings via %s (single batched call)",
            llm.settings.embed_model,
        )
        embeddings: list[list[float] | None] = list(llm.embed(texts))  # type: ignore[arg-type]
    else:
        log.warning(
            "ingest: no OPENAI_API_KEY -- inserting zero-vector embeddings (BM25-only mode)"
        )
        dim = llm.settings.embedding_dim
        embeddings = [[0.0] * dim for _ in records]

    rows = [_row_from_record(rec, emb) for rec, emb in zip(records, embeddings, strict=True)]
    session.add_all(rows)
    session.commit()

    log.info("ingest: inserted %d jobs", len(rows))
    return len(rows)
