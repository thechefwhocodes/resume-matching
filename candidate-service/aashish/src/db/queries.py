"""Read-side queries against the jobs table.

`retrieve_with_filters` is the P1c entry point used by the embed+filters and
embed+rerank pipelines. It runs:

  1. pgvector cosine retrieval to top-N candidates
  2. SQL hard filters (status='active', H1B if needed, severe YoE under-qual)
  3. Python-side soft scoring (category/YoE/location bonuses)
  4. Re-rank by `final_score = sim + soft_bonuses`
  5. Returns ranked rows + a per-row `filter_flags` dict for explainability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.models import Job

if TYPE_CHECKING:
    from src.extractor.prompts import ResumeProfile

CATEGORY_BONUS = 0.10
YOE_ALIGN_BONUS = 0.05
LOCATION_CITY_BONUS = 0.05
LOCATION_REMOTE_BONUS = 0.03
LOCATION_BONUS_CAP = 0.05

YOE_OVER_QUAL_TOLERANCE = 2  # job.yoe_min <= candidate_yoe + 2 -> still considered


def list_active_jobs(session: Session, limit: int | None = None) -> list[Job]:
    stmt = select(Job).where(Job.status == "active").order_by(Job.inserted_at)
    if limit is not None:
        stmt = stmt.limit(limit)
    return list(session.execute(stmt).scalars())


def count_jobs(session: Session) -> int:
    return session.query(Job).count()


def get_job(session: Session, job_id: str) -> Job | None:
    return session.get(Job, job_id)


@dataclass(slots=True)
class FilteredCandidate:
    job: Job
    sim: float
    final_score: float
    filter_flags: dict = field(default_factory=dict)


def _location_bonus(job: Job, profile: ResumeProfile) -> tuple[float, str | None]:
    """Return (bonus, reason). Capped at LOCATION_BONUS_CAP per design."""
    job_loc = (job.location or "").strip().lower()
    cand_loc = (profile.location_city or "").strip().lower()
    if job_loc and cand_loc and (cand_loc in job_loc or job_loc in cand_loc):
        return min(LOCATION_CITY_BONUS, LOCATION_BONUS_CAP), "exact_city"

    work_type = (job.work_location_type or "").lower()
    if profile.open_to_remote and work_type in ("remote", "hybrid"):
        return min(LOCATION_REMOTE_BONUS, LOCATION_BONUS_CAP), "remote_friendly"

    return 0.0, None


def _category_bonus(job: Job, profile: ResumeProfile) -> tuple[float, str | None]:
    cat = job.job_category or ""
    if cat == profile.primary_category:
        return CATEGORY_BONUS, "primary_category"
    if cat in profile.secondary_categories:
        return round(CATEGORY_BONUS / 2.0, 4), "secondary_category"
    return 0.0, None


def _yoe_bonus(job: Job, profile: ResumeProfile) -> tuple[float, str | None]:
    job_yoe = job.yoe_min if job.yoe_min is not None else 0
    cand = profile.years_experience
    if job_yoe == 0:
        return 0.0, None
    diff = cand - job_yoe
    if -1 <= diff <= 3:
        return YOE_ALIGN_BONUS, "yoe_aligned"
    return 0.0, None


def _hard_filter_keep(job: Job, profile: ResumeProfile) -> tuple[bool, dict]:
    """Return (keep, reasons_dropped_or_kept)."""
    flags: dict = {}

    if (job.status or "").lower() != "active":
        return False, {"dropped": "inactive"}

    if profile.needs_h1b_sponsorship and not job.h1b_sponsorship:
        return False, {"dropped": "no_h1b_sponsorship"}

    job_yoe = job.yoe_min or 0
    if job_yoe > profile.years_experience + YOE_OVER_QUAL_TOLERANCE:
        return False, {
            "dropped": "yoe_under_qualified",
            "job_yoe_min": job_yoe,
            "candidate_yoe": profile.years_experience,
        }

    if profile.needs_h1b_sponsorship:
        flags["h1b_ok"] = True
    return True, flags


def filter_and_score(
    candidates: list[tuple[Job, float]],
    profile: ResumeProfile,
) -> list[FilteredCandidate]:
    """Apply hard filters + soft scoring to a list of (job, similarity) pairs.

    Returns the surviving candidates sorted by descending final_score.
    """
    out: list[FilteredCandidate] = []
    for job, sim in candidates:
        keep, hard_flags = _hard_filter_keep(job, profile)
        if not keep:
            continue
        flags = dict(hard_flags)

        cat_bonus, cat_reason = _category_bonus(job, profile)
        yoe_bonus, yoe_reason = _yoe_bonus(job, profile)
        loc_bonus, loc_reason = _location_bonus(job, profile)

        if cat_reason:
            flags["category"] = cat_reason
        if yoe_reason:
            flags["yoe"] = yoe_reason
        if loc_reason:
            flags["location"] = loc_reason

        final = sim + cat_bonus + yoe_bonus + loc_bonus
        flags["bonuses"] = {
            "category": round(cat_bonus, 4),
            "yoe": round(yoe_bonus, 4),
            "location": round(loc_bonus, 4),
        }
        flags["sim"] = round(float(sim), 4)
        flags["final_score"] = round(float(final), 4)

        out.append(FilteredCandidate(job=job, sim=float(sim), final_score=float(final), filter_flags=flags))

    out.sort(key=lambda c: c.final_score, reverse=True)
    return out


def retrieve_with_filters(
    session: Session,
    *,
    candidate_pairs: list[tuple[str, float]],
    profile: ResumeProfile,
) -> list[FilteredCandidate]:
    """Hydrate (job_id, sim) pairs into Jobs and apply hard+soft filters.

    The retrieval step lives in the embedding retriever; this function is
    the SQL+Python "WHERE/ORDER BY" stage.
    """
    if not candidate_pairs:
        return []

    ids = [jid for jid, _ in candidate_pairs]
    rows = session.execute(select(Job).where(Job.job_id.in_(ids))).scalars().all()
    by_id = {r.job_id: r for r in rows}
    pairs: list[tuple[Job, float]] = []
    for jid, sim in candidate_pairs:
        job = by_id.get(jid)
        if job is not None:
            pairs.append((job, sim))

    return filter_and_score(pairs, profile)
