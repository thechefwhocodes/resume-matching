"""Pydantic schema roundtrip + contract sanity."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.schemas import (
    ApproachName,
    JobMatch,
    MatchMetadata,
    MatchRequest,
    MatchResponse,
    Resume,
    TokenUsage,
)


def test_match_request_minimal_is_valid():
    req = MatchRequest(resume=Resume(content="hello world"))
    assert req.approach is None
    assert req.resume.content == "hello world"


def test_match_request_with_approach():
    req = MatchRequest(
        resume=Resume(content="x"),
        approach="embed+rerank",
    )
    assert req.approach == "embed+rerank"


def test_match_request_rejects_bad_approach():
    with pytest.raises(ValidationError):
        MatchRequest.model_validate({"resume": {"content": "x"}, "approach": "magic"})


def test_match_response_roundtrip():
    resp = MatchResponse(
        matches=[
            JobMatch(
                job_id="abc",
                title="Backend Engineer",
                company="Stripe",
                match_score=87.5,
                explanation="Strong overlap",
                matching_skills=["Go", "Postgres"],
                experience_alignment="Senior",
                location="Seattle, WA",
                salary_range="$180K - $240K",
                job_category="Backend",
                retrieval_score=0.81,
                rerank_score=87.5,
                filter_flags={"category_bonus": 0.1},
            )
        ],
        metadata=MatchMetadata(
            retrieval_method="embed",
            reranking_method="llm-rerank",
            processing_time_ms=4321,
            retrieval_count=30,
            filtered_count=22,
            returned_count=1,
            approach="embed+rerank",
            cost_usd=0.0042,
            tokens=TokenUsage(prompt=1200, completion=300, embedding=128),
        ),
    )

    payload = resp.model_dump()
    restored = MatchResponse.model_validate(payload)
    assert restored.metadata.approach == "embed+rerank"
    assert restored.matches[0].salary_range == "$180K - $240K"
    assert restored.metadata.tokens.prompt == 1200


def test_jobmatch_score_bounds_enforced():
    with pytest.raises(ValidationError):
        JobMatch(
            job_id="x",
            title="t",
            company="c",
            match_score=150,
            explanation="",
        )


def test_approach_name_literal_values():
    valid: list[ApproachName] = ["bm25", "bm25+rerank", "embed", "embed+rerank"]
    assert len(set(valid)) == 4
