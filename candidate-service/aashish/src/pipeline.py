"""Approach selector + per-approach pipeline implementations.

Approaches:
    bm25            -- in-memory BM25 only
    bm25+rerank     -- BM25 retrieve, LLM rerank        (P1d)
    embed           -- pgvector cosine retrieve only    (P1b)
    embed+rerank    -- embed + filters + rerank         (P1b/P1c/P1d)

This module is the single dispatch point used by main.py. Each phase replaces
one branch; everything else continues to work.
"""

from __future__ import annotations

import logging
from time import perf_counter

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.config import ApproachName, get_settings
from src.db.models import Job
from src.db.queries import get_job, retrieve_with_filters
from src.extractor.extractor import extract_resume_profile
from src.extractor.prompts import ResumeProfile
from src.llm.client import CostTracker, LLMClient
from src.rerank.reranker import RerankOutcome, rerank
from src.retrieval.base import Retriever
from src.retrieval.embeddings import EmbeddingRetriever
from src.schemas import (
    JobMatch,
    MatchMetadata,
    MatchRequest,
    MatchResponse,
    TokenUsage,
)
from src.utils.salary import format_salary_range

log = logging.getLogger(__name__)


def select_approach(request: MatchRequest, *, llm: LLMClient) -> ApproachName:
    """Pick the approach for this request.

    Precedence: explicit body field > APPROACH_OVERRIDE env > LLM-availability default.
    """
    if request.approach is not None:
        return request.approach
    return llm.settings.default_approach()


def run_pipeline(
    request: MatchRequest,
    *,
    session: Session,
    llm: LLMClient,
    bm25_index: Retriever | None,
) -> MatchResponse:
    """Entry point used by the FastAPI handler."""
    settings = get_settings()
    approach = select_approach(request, llm=llm)
    started = perf_counter()
    tracker = CostTracker()

    if approach == "bm25":
        matches, retrieval_count, filtered_count = _bm25_pipeline(
            request,
            session=session,
            bm25_index=bm25_index,
            top_k=settings.result_top_k,
        )
    elif approach == "bm25+rerank":
        matches, retrieval_count, filtered_count = _bm25_rerank_pipeline(
            request,
            session=session,
            bm25_index=bm25_index,
            llm=llm,
            tracker=tracker,
            retrieval_top_k=settings.retrieval_top_k,
            result_top_k=settings.result_top_k,
            min_score=settings.rerank_min_score,
        )
    elif approach == "embed":
        matches, retrieval_count, filtered_count = _embed_pipeline(
            request,
            session=session,
            llm=llm,
            tracker=tracker,
            bm25_index=bm25_index,
            retrieval_top_k=settings.retrieval_top_k,
            result_top_k=settings.result_top_k,
        )
    elif approach == "embed+rerank":
        matches, retrieval_count, filtered_count = _embed_rerank_pipeline(
            request,
            session=session,
            llm=llm,
            tracker=tracker,
            bm25_index=bm25_index,
            retrieval_top_k=settings.retrieval_top_k,
            result_top_k=settings.result_top_k,
            min_score=settings.rerank_min_score,
        )
    else:
        raise ValueError(f"Unknown approach: {approach}")

    elapsed_ms = int((perf_counter() - started) * 1000)
    metadata = MatchMetadata(
        retrieval_method=_retrieval_method_label(approach),
        reranking_method=_rerank_method_label(approach),
        processing_time_ms=elapsed_ms,
        retrieval_count=retrieval_count,
        filtered_count=filtered_count,
        returned_count=len(matches),
        approach=approach,
        cost_usd=round(tracker.cost_usd, 6),
        tokens=TokenUsage(
            prompt=tracker.prompt,
            completion=tracker.completion,
            embedding=tracker.embedding,
        ),
    )
    return MatchResponse(matches=matches, metadata=metadata)


def _retrieval_method_label(approach: ApproachName) -> str:
    if approach.startswith("embed"):
        return "embed"
    return "bm25"


def _rerank_method_label(approach: ApproachName) -> str:
    return "llm-rerank" if approach.endswith("+rerank") else "none"


# ---------------------------------------------------------------------------
# BM25 pipeline (P1a)
# ---------------------------------------------------------------------------


def _bm25_pipeline(
    request: MatchRequest,
    *,
    session: Session,
    bm25_index: Retriever | None,
    top_k: int,
) -> tuple[list[JobMatch], int, int]:
    if bm25_index is None:
        log.error("bm25 pipeline called without an index; returning empty result")
        return [], 0, 0

    pairs = bm25_index.retrieve(request.resume.content, top_k=top_k)
    matches: list[JobMatch] = []

    if not pairs:
        return [], 0, 0

    max_score = max(score for _, score in pairs) or 1.0

    for job_id, score in pairs:
        job = get_job(session, job_id)
        if job is None:
            continue
        normalized = round((score / max_score) * 100.0, 2)
        matches.append(_to_jobmatch(
            job,
            match_score=normalized,
            retrieval_score=float(score),
            explanation=(
                f"BM25 lexical match on title/responsibilities/requirements "
                f"(raw score {score:.2f}, normalized {normalized:.0f}%)."
            ),
        ))

    # No filter stage in BM25, so filtered_count == retrieval_count.
    # returned_count (set by run_pipeline) is len(matches), which may be
    # smaller if BM25 returned a job_id whose row was purged after index build.
    return matches, len(pairs), len(pairs)


# ---------------------------------------------------------------------------
# Embedding pipeline (P1b)
# ---------------------------------------------------------------------------


def _embed_pipeline(
    request: MatchRequest,
    *,
    session: Session,
    llm: LLMClient,
    tracker: CostTracker,
    bm25_index: Retriever | None,
    retrieval_top_k: int,
    result_top_k: int,
) -> tuple[list[JobMatch], int, int]:
    if not llm.is_available:
        log.warning("embed pipeline requested without OPENAI_API_KEY; falling back to BM25")
        return _bm25_pipeline(
            request, session=session, bm25_index=bm25_index, top_k=result_top_k
        )

    retriever = EmbeddingRetriever(llm=llm, session=session, tracker=tracker)
    pairs = retriever.retrieve(request.resume.content, top_k=retrieval_top_k)

    if not pairs:
        return [], 0, 0

    top = pairs[:result_top_k]
    matches: list[JobMatch] = []
    for job_id, sim in top:
        job = get_job(session, job_id)
        if job is None:
            continue
        score_pct = round(max(0.0, min(1.0, (sim + 1.0) / 2.0)) * 100.0, 2)
        matches.append(
            _to_jobmatch(
                job,
                match_score=score_pct,
                retrieval_score=float(sim),
                explanation=(
                    f"Dense semantic match via {llm.settings.embed_model} "
                    f"(cosine similarity {sim:.3f})."
                ),
            )
        )

    return matches, len(pairs), len(top)


def _job_text_mentions(job: Job, term: str) -> bool:
    if not term:
        return False
    needle = term.strip().lower()
    if not needle:
        return False
    haystack = " ".join(
        s for s in (job.title, job.requirements, job.responsibilities_text) if s
    ).lower()
    return needle in haystack


# ---------------------------------------------------------------------------
# Rerank pipelines (P1d)
# ---------------------------------------------------------------------------


def _bm25_rerank_pipeline(
    request: MatchRequest,
    *,
    session: Session,
    bm25_index: Retriever | None,
    llm: LLMClient,
    tracker: CostTracker,
    retrieval_top_k: int,
    result_top_k: int,
    min_score: int,
) -> tuple[list[JobMatch], int, int]:
    if bm25_index is None:
        return [], 0, 0
    if not llm.is_available:
        log.warning("bm25+rerank requested without OPENAI_API_KEY; falling back to BM25")
        return _bm25_pipeline(
            request, session=session, bm25_index=bm25_index, top_k=result_top_k
        )

    pairs = bm25_index.retrieve(request.resume.content, top_k=retrieval_top_k)
    if not pairs:
        return [], 0, 0

    profile = extract_resume_profile(
        session,
        content=request.resume.content,
        llm=llm,
        tracker=tracker,
        filename=request.resume.filename,
    )

    ids = [jid for jid, _ in pairs]
    rows = session.execute(select(Job).where(Job.job_id.in_(ids))).scalars().all()
    by_id = {r.job_id: r for r in rows}
    candidates = [by_id[jid] for jid, _ in pairs if jid in by_id]
    sim_lookup = {jid: float(score) for jid, score in pairs}

    outcome = rerank(
        resume_text=request.resume.content,
        profile=profile,
        candidates=candidates,
        llm=llm,
        tracker=tracker,
    )

    return _build_rerank_matches(
        candidates=candidates,
        sim_lookup=sim_lookup,
        outcome=outcome,
        profile=profile,
        retrieval_method_label="bm25",
        min_score=min_score,
        result_top_k=result_top_k,
    )


def _embed_rerank_pipeline(
    request: MatchRequest,
    *,
    session: Session,
    llm: LLMClient,
    tracker: CostTracker,
    bm25_index: Retriever | None,
    retrieval_top_k: int,
    result_top_k: int,
    min_score: int,
) -> tuple[list[JobMatch], int, int]:
    if not llm.is_available:
        log.warning("embed+rerank requested without OPENAI_API_KEY; falling back to BM25")
        return _bm25_pipeline(
            request, session=session, bm25_index=bm25_index, top_k=result_top_k
        )

    profile = extract_resume_profile(
        session,
        content=request.resume.content,
        llm=llm,
        tracker=tracker,
        filename=request.resume.filename,
    )

    retriever = EmbeddingRetriever(llm=llm, session=session, tracker=tracker)
    pairs = retriever.retrieve(request.resume.content, top_k=retrieval_top_k)
    if not pairs:
        return [], 0, 0

    survivors = retrieve_with_filters(
        session,
        candidate_pairs=pairs,
        profile=profile,
    )
    if not survivors:
        return [], len(pairs), 0

    candidates = [c.job for c in survivors]
    sim_lookup = {c.job.job_id: float(c.sim) for c in survivors}
    flag_lookup = {c.job.job_id: c.filter_flags for c in survivors}

    outcome = rerank(
        resume_text=request.resume.content,
        profile=profile,
        candidates=candidates,
        llm=llm,
        tracker=tracker,
    )

    return _build_rerank_matches(
        candidates=candidates,
        sim_lookup=sim_lookup,
        outcome=outcome,
        profile=profile,
        retrieval_method_label="embed+filters",
        min_score=min_score,
        result_top_k=result_top_k,
        retrieval_count=len(pairs),
        filter_flag_lookup=flag_lookup,
    )


def _build_rerank_matches(
    *,
    candidates: list[Job],
    sim_lookup: dict[str, float],
    outcome: RerankOutcome,
    profile: ResumeProfile,
    retrieval_method_label: str,
    min_score: int,
    result_top_k: int,
    retrieval_count: int | None = None,
    filter_flag_lookup: dict[str, dict] | None = None,
) -> tuple[list[JobMatch], int, int]:
    items = outcome.items_by_job_id
    matches: list[JobMatch] = []
    for job in candidates:
        item = items.get(job.job_id)
        if item is None:
            continue
        if item.fit_score < min_score:
            continue

        skills_overlap = sorted(
            s for s in profile.skills if _job_text_mentions(job, s)
        )[:8]
        explanation = (item.explanation or " ".join(item.reasons))[:240]
        if not explanation:
            explanation = "LLM rerank judged this a match."

        flags = dict(filter_flag_lookup.get(job.job_id, {})) if filter_flag_lookup else {}
        flags["rerank"] = {
            "fit_score": item.fit_score,
            "reasons": item.reasons,
            "concerns": item.concerns,
        }

        matches.append(
            _to_jobmatch(
                job,
                match_score=float(item.fit_score),
                retrieval_score=sim_lookup.get(job.job_id),
                rerank_score=float(item.fit_score),
                explanation=explanation,
                matching_skills=skills_overlap,
                experience_alignment=(
                    f"Job wants {job.yoe_min or 0}+ yrs; candidate has "
                    f"{profile.years_experience:g}."
                ),
                filter_flags=flags or None,
            )
        )

    matches.sort(key=lambda m: m.rerank_score or 0.0, reverse=True)
    matches = matches[:result_top_k]

    retrieval_total = retrieval_count if retrieval_count is not None else len(candidates)
    return matches, retrieval_total, len(matches)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _to_jobmatch(
    job: Job,
    *,
    match_score: float,
    retrieval_score: float | None = None,
    rerank_score: float | None = None,
    explanation: str = "",
    matching_skills: list[str] | None = None,
    experience_alignment: str = "",
    filter_flags: dict | None = None,
) -> JobMatch:
    salary_range = job.salary_range or format_salary_range(
        job.salary_min, job.salary_max, job.salary_currency
    )
    return JobMatch(
        job_id=job.job_id,
        title=job.title,
        company=job.company_name,
        match_score=match_score,
        explanation=explanation,
        matching_skills=matching_skills or [],
        experience_alignment=experience_alignment,
        location=job.location,
        salary_range=salary_range,
        job_category=job.job_category,
        responsibilities=job.responsibilities_text,
        requirements=job.requirements,
        retrieval_score=retrieval_score,
        rerank_score=rerank_score,
        filter_flags=filter_flags,
    )
