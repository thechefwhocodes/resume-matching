"""Pydantic schemas mirroring the TypeScript contract in src/types/index.ts.

These cross the wire to the Next.js proxy, so field names must stay aligned.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ApproachName = Literal["bm25", "bm25+rerank", "embed", "embed+rerank"]


class Resume(BaseModel):
    model_config = ConfigDict(extra="ignore")

    content: str
    filename: str | None = None
    format: Literal["txt"] | None = "txt"


class MatchRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    resume: Resume
    approach: ApproachName | None = Field(
        default=None,
        description=(
            "Optional approach override. When omitted, the service picks "
            "embed+rerank if OPENAI_API_KEY is set, otherwise bm25."
        ),
    )


class JobMatch(BaseModel):
    model_config = ConfigDict(extra="allow")

    job_id: str
    title: str
    company: str
    match_score: float = Field(ge=0, le=100)
    explanation: str
    matching_skills: list[str] = Field(default_factory=list)
    experience_alignment: str = ""

    location: str | None = None
    salary_range: str | None = None
    job_category: str | None = None
    responsibilities: str | None = None
    requirements: str | None = None

    retrieval_score: float | None = None
    rerank_score: float | None = None
    filter_flags: dict[str, Any] | None = None


class TokenUsage(BaseModel):
    prompt: int = 0
    completion: int = 0
    embedding: int = 0


class MatchMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    retrieval_method: str
    reranking_method: str
    processing_time_ms: int

    retrieval_count: int = 0
    filtered_count: int = 0
    returned_count: int = 0
    approach: ApproachName
    cost_usd: float = 0.0
    tokens: TokenUsage = Field(default_factory=TokenUsage)


class MatchResponse(BaseModel):
    matches: list[JobMatch]
    metadata: MatchMetadata


class ErrorResponse(BaseModel):
    error: str
    message: str
    details: Any | None = None
