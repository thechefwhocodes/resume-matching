"""Resume profile extractor with sha256-keyed Postgres cache.

The extractor is a single LLM call (gpt-4o-mini, structured outputs). The
result is cached in `resume_profile_cache` keyed by (sha256(resume), model)
so repeated calls for the same resume are free. The Resume table separately
stores the raw text under the same sha256.
"""

from __future__ import annotations

import hashlib
import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.models import Resume, ResumeProfileCache
from src.extractor.prompts import (
    SYSTEM_PROMPT,
    ResumeProfile,
    build_user_prompt,
)
from src.llm.client import CostTracker, LLMClient

log = logging.getLogger(__name__)


def resume_sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def upsert_resume(
    session: Session,
    *,
    content: str,
    filename: str | None = None,
) -> str:
    sha = resume_sha256(content)
    existing = session.get(Resume, sha)
    if existing is None:
        session.add(Resume(sha256=sha, content=content, filename=filename))
        session.commit()
    return sha


def extract_resume_profile(
    session: Session,
    *,
    content: str,
    llm: LLMClient,
    tracker: CostTracker,
    filename: str | None = None,
) -> ResumeProfile:
    """Return a ResumeProfile, hitting cache if available else calling the LLM."""
    sha = upsert_resume(session, content=content, filename=filename)
    model = llm.settings.extract_model

    cached = session.execute(
        select(ResumeProfileCache).where(
            ResumeProfileCache.sha256 == sha,
            ResumeProfileCache.model == model,
        )
    ).scalar_one_or_none()
    if cached is not None:
        log.info("extractor: cache hit for sha=%s model=%s", sha[:12], model)
        return ResumeProfile.model_validate(cached.profile_json)

    if not llm.is_available:
        raise RuntimeError(
            "extractor: cannot extract profile without OPENAI_API_KEY. "
            "Pipelines that need a profile must guard with llm.is_available."
        )

    log.info("extractor: cache miss for sha=%s model=%s; calling LLM", sha[:12], model)
    profile = llm.chat_structured(
        response_model=ResumeProfile,
        system=SYSTEM_PROMPT,
        user=build_user_prompt(content),
        model=model,
        temperature=0.0,
        seed=42,
        tracker=tracker,
    )

    session.merge(
        ResumeProfileCache(
            sha256=sha,
            model=model,
            profile_json=profile.model_dump(),
        )
    )
    session.commit()

    return profile
