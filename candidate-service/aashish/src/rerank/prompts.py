"""Single-batched LLM rerank prompt + structured output schema.

The reranker scores all 30 candidates in one call. We deliberately put the
full job context (greenFlags / redFlags / idealCompanies) in the prompt so the
model can use it -- those signals don't make it into the embedding.

The model returns a list of `RerankItem` keyed by job_id with a 0-100 fit
score, top 3 reasons, and a 1-line explanation. The pipeline drops anything
below `RERANK_MIN_SCORE` (default 50).
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

RERANK_PROMPT_VERSION = "v1"


class RerankItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    job_id: str = Field(description="The job_id from the candidate list, copied verbatim.")
    fit_score: float = Field(
        ge=0,
        le=100,
        description=(
            "Calibrated 0-100 score. Use the rubric:\n"
            "  90-100  Exceptional fit; we'd interview immediately.\n"
            "  75-89   Strong fit on most axes; minor gaps.\n"
            "  60-74   Reasonable fit; some real concerns.\n"
            "  40-59   Weak fit; major mismatches in seniority/skills/category.\n"
            "  0-39    Wrong category, wrong seniority, or hard mismatch (e.g. needs visa we don't sponsor)."
        ),
    )
    reasons: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="Top 3 short bullets describing the strongest match signals.",
    )
    concerns: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="Up to 3 concerns or gaps. Empty if none.",
    )
    explanation: str = Field(
        default="",
        max_length=240,
        description="One-sentence summary suitable for showing to the candidate.",
    )


class RerankResults(BaseModel):
    items: list[RerankItem]


SYSTEM_PROMPT = (
    "You are a senior tech recruiter scoring candidate-job fit. Be calibrated "
    "and conservative: most candidates are mediocre fits and should score "
    "40-70. Only top-decile fits earn 85+. Use the green/red flags and "
    "ideal-companies for each job to refine your judgement. Never invent "
    "facts not in the resume or the job description. Return one entry per "
    "candidate id, in any order."
)


def _job_block(job: dict[str, Any]) -> str:
    parts = [
        f"job_id: {job['job_id']}",
        f"title: {job.get('title','')}  @  {job.get('company','')}",
        f"category: {job.get('job_category','')}  yoe_min: {job.get('yoe_min','?')}  "
        f"location: {job.get('location','')}  type: {job.get('work_location_type','')}  "
        f"h1b: {bool(job.get('h1b_sponsorship', False))}",
    ]
    requirements = (job.get("requirements") or "").strip()
    if requirements:
        parts.append(f"requirements: {requirements[:600]}")
    responsibilities = (job.get("responsibilities") or "").strip()
    if responsibilities:
        parts.append(f"responsibilities: {responsibilities[:600]}")
    if job.get("greenFlags"):
        parts.append(f"greenFlags: {json.dumps(job['greenFlags'])}")
    if job.get("redFlags"):
        parts.append(f"redFlags: {json.dumps(job['redFlags'])}")
    if job.get("idealCompanies"):
        parts.append(f"idealCompanies: {json.dumps(job['idealCompanies'])}")
    return "\n".join(parts)


def build_user_prompt(
    *,
    resume_text: str,
    profile_summary: str,
    candidates: list[dict[str, Any]],
) -> str:
    candidate_blocks = "\n\n---\n\n".join(_job_block(c) for c in candidates)
    profile_line = f"PROFILE: {profile_summary}\n\n" if profile_summary else ""
    return (
        "Score each candidate job 0-100 for the resume below.\n\n"
        f"{profile_line}"
        "----- RESUME START -----\n"
        f"{resume_text.strip()[:4000]}\n"
        "----- RESUME END -----\n\n"
        f"CANDIDATE JOBS ({len(candidates)} total):\n\n"
        f"{candidate_blocks}\n"
    )
