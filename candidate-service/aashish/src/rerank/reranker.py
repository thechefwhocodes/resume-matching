"""LLM reranker: one batched chat_structured call over all 30 candidates."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from src.db.models import Job
from src.extractor.prompts import ResumeProfile
from src.llm.client import CostTracker, LLMClient
from src.rerank.prompts import (
    SYSTEM_PROMPT,
    RerankItem,
    RerankResults,
    build_user_prompt,
)

log = logging.getLogger(__name__)


@dataclass(slots=True)
class RerankOutcome:
    items_by_job_id: dict[str, RerankItem]


def _job_to_prompt_dict(job: Job) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "title": job.title,
        "company": job.company_name,
        "job_category": job.job_category,
        "yoe_min": job.yoe_min,
        "location": job.location,
        "work_location_type": job.work_location_type,
        "h1b_sponsorship": job.h1b_sponsorship,
        "requirements": job.requirements,
        "responsibilities": job.responsibilities_text,
        "greenFlags": job.green_flags,
        "redFlags": job.red_flags,
        "idealCompanies": job.ideal_companies,
    }


def rerank(
    *,
    resume_text: str,
    profile: ResumeProfile,
    candidates: list[Job],
    llm: LLMClient,
    tracker: CostTracker,
) -> RerankOutcome:
    """Issue one rerank call. Returns a dict of `{job_id: RerankItem}`.

    On structural failure (e.g. model refuses or returns no items), the
    caller should fall back to the pre-rerank order.
    """
    if not candidates:
        return RerankOutcome(items_by_job_id={})
    if not llm.is_available:
        raise RuntimeError("rerank() called without OPENAI_API_KEY")

    user_prompt = build_user_prompt(
        resume_text=resume_text,
        profile_summary=profile.summary,
        candidates=[_job_to_prompt_dict(j) for j in candidates],
    )

    log.info(
        "rerank: calling %s on %d candidates", llm.settings.rerank_model, len(candidates)
    )
    parsed: RerankResults = llm.chat_structured(
        response_model=RerankResults,
        system=SYSTEM_PROMPT,
        user=user_prompt,
        model=llm.settings.rerank_model,
        temperature=0.0,
        seed=42,
        tracker=tracker,
    )

    valid_ids = {j.job_id for j in candidates}
    by_id: dict[str, RerankItem] = {}
    for item in parsed.items:
        if item.job_id in valid_ids:
            by_id[item.job_id] = item
        else:
            log.debug("rerank: model returned unknown job_id=%s; ignoring", item.job_id)

    log.info("rerank: parsed %d/%d valid scores", len(by_id), len(candidates))
    return RerankOutcome(items_by_job_id=by_id)
