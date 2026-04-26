"""Smoke-check the rerank prompt builder + RerankResults schema parsing."""

from __future__ import annotations

from src.rerank.prompts import RerankItem, RerankResults, build_user_prompt


def test_build_user_prompt_contains_resume_and_jobs():
    p = build_user_prompt(
        resume_text="Senior Backend Engineer with 8 years of experience.",
        profile_summary="Senior backend engineer, US-based, no sponsorship needed.",
        candidates=[
            {
                "job_id": "abc",
                "title": "Backend Engineer",
                "company": "Stripe",
                "job_category": "Backend",
                "yoe_min": 5,
                "location": "San Francisco, CA",
                "work_location_type": "Hybrid",
                "h1b_sponsorship": True,
                "requirements": "5+ years Go, gRPC, Postgres",
                "responsibilities": "Build payment infra.",
                "greenFlags": ["Strong systems thinking"],
                "redFlags": ["Job hopping"],
                "idealCompanies": ["Square", "Block"],
            }
        ],
    )
    assert "abc" in p
    assert "Backend Engineer" in p
    assert "Stripe" in p
    assert "greenFlags" in p
    assert "redFlags" in p
    assert "Senior Backend Engineer" in p
    assert "PROFILE:" in p


def test_rerank_results_validates_score_bounds():
    parsed = RerankResults.model_validate(
        {
            "items": [
                {
                    "job_id": "abc",
                    "fit_score": 87.5,
                    "reasons": ["Go expertise", "Distributed systems"],
                    "concerns": [],
                    "explanation": "Strong fit on stack and seniority.",
                }
            ]
        }
    )
    assert parsed.items[0].fit_score == 87.5
    assert isinstance(parsed.items[0], RerankItem)


def test_rerank_results_clamps_invalid_score():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RerankResults.model_validate(
            {"items": [{"job_id": "x", "fit_score": 150, "reasons": [], "explanation": ""}]}
        )
