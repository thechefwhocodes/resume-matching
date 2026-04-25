"""ORM models. All persistence lives in Postgres; pgvector handles dense search.

Tables:
  - jobs                 ingested from data/jobs.json on first boot
  - resumes              raw text cache, keyed by sha256(content)
  - resume_profile_cache extracted ResumeProfile JSON, 1:1 with resumes
  - ground_truth         hand-curated job-id picks per resume (eval only)
  - eval_runs            one row per `make eval` invocation
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from src.config import get_settings
from src.db.engine import Base

_DIM = get_settings().embedding_dim


class Job(Base):
    __tablename__ = "jobs"

    job_id: Mapped[str] = mapped_column(String, primary_key=True)
    title: Mapped[str] = mapped_column(String, nullable=False)
    company_name: Mapped[str] = mapped_column(String, nullable=False)
    company_id: Mapped[str | None] = mapped_column(String, nullable=True)

    responsibilities_html: Mapped[str | None] = mapped_column(Text, nullable=True)
    responsibilities_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    requirements: Mapped[str | None] = mapped_column(Text, nullable=True)

    job_category: Mapped[str | None] = mapped_column(String, nullable=True)
    location: Mapped[str | None] = mapped_column(String, nullable=True)
    work_location_type: Mapped[str | None] = mapped_column(String, nullable=True)

    yoe_min: Mapped[int | None] = mapped_column(Integer, nullable=True)
    yoe_label: Mapped[str | None] = mapped_column(String, nullable=True)

    salary_min: Mapped[float | None] = mapped_column(Float, nullable=True)
    salary_max: Mapped[float | None] = mapped_column(Float, nullable=True)
    salary_currency: Mapped[str | None] = mapped_column(String, nullable=True)
    salary_range: Mapped[str | None] = mapped_column(String, nullable=True)

    equity_min: Mapped[float | None] = mapped_column(Float, nullable=True)
    equity_max: Mapped[float | None] = mapped_column(Float, nullable=True)

    employment_type: Mapped[str | None] = mapped_column(String, nullable=True)
    benefits: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)

    h1b_sponsorship: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="active")
    created_at: Mapped[str | None] = mapped_column(String, nullable=True)

    green_flags: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    red_flags: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    ideal_companies: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)

    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding = mapped_column(Vector(_DIM), nullable=True)

    inserted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class Resume(Base):
    __tablename__ = "resumes"

    sha256: Mapped[str] = mapped_column(String(64), primary_key=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    filename: Mapped[str | None] = mapped_column(String, nullable=True)
    inserted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class ResumeProfileCache(Base):
    __tablename__ = "resume_profile_cache"

    sha256: Mapped[str] = mapped_column(String(64), primary_key=True)
    model: Mapped[str] = mapped_column(String, nullable=False)
    profile_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    cached_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (UniqueConstraint("sha256", "model", name="uq_resume_profile_sha_model"),)


class GroundTruth(Base):
    __tablename__ = "ground_truth"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    resume_file: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    expected_job_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(String, nullable=False, default="claude-opus-4")
    inserted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class EvalRun(Base):
    __tablename__ = "eval_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    suite: Mapped[str] = mapped_column(String, nullable=False)
    approach: Mapped[str | None] = mapped_column(String, nullable=True)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
