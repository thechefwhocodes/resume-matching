"""Hard SQL filter + soft scoring logic, tested without a database."""

from __future__ import annotations

from src.db.models import Job
from src.db.queries import filter_and_score
from src.extractor.prompts import ResumeProfile


def _job(
    *,
    job_id: str,
    category: str = "Backend",
    yoe_min: int = 5,
    h1b: bool = True,
    location: str = "Seattle, WA",
    work_location_type: str = "Hybrid",
    status: str = "active",
) -> Job:
    return Job(
        job_id=job_id,
        title=f"Job {job_id}",
        company_name="C",
        job_category=category,
        yoe_min=yoe_min,
        h1b_sponsorship=h1b,
        location=location,
        work_location_type=work_location_type,
        status=status,
        content_sha256="0" * 64,
    )


def _profile(**overrides) -> ResumeProfile:
    base = dict(
        years_experience=8.0,
        primary_category="Backend",
        secondary_categories=["DevOps"],
        seniority="Senior",
        needs_h1b_sponsorship=False,
        open_to_remote=False,
        location_city="Seattle, WA",
        skills=["go", "python"],
        summary="",
    )
    base.update(overrides)
    return ResumeProfile.model_validate(base)


def test_drops_inactive():
    j = _job(job_id="x", status="closed")
    out = filter_and_score([(j, 0.8)], _profile())
    assert out == []


def test_drops_when_h1b_required_but_unsupported():
    j = _job(job_id="x", h1b=False)
    out = filter_and_score([(j, 0.8)], _profile(needs_h1b_sponsorship=True))
    assert out == []


def test_keeps_when_h1b_supported():
    j = _job(job_id="x", h1b=True)
    out = filter_and_score([(j, 0.8)], _profile(needs_h1b_sponsorship=True))
    assert len(out) == 1
    assert out[0].filter_flags.get("h1b_ok") is True


def test_drops_severe_yoe_under_qual():
    j = _job(job_id="x", yoe_min=12)
    out = filter_and_score([(j, 0.8)], _profile(years_experience=5.0))
    assert out == []


def test_keeps_within_yoe_tolerance():
    j = _job(job_id="x", yoe_min=7)
    out = filter_and_score([(j, 0.8)], _profile(years_experience=5.0))
    assert len(out) == 1


def test_primary_category_bonus_applied():
    j = _job(job_id="x", category="Backend")
    out = filter_and_score([(j, 0.5)], _profile())
    flags = out[0].filter_flags
    assert flags["category"] == "primary_category"
    assert flags["bonuses"]["category"] == 0.10
    assert out[0].final_score > 0.5


def test_secondary_category_bonus_smaller():
    j = _job(job_id="x", category="DevOps")
    out = filter_and_score([(j, 0.5)], _profile())
    assert out[0].filter_flags["category"] == "secondary_category"
    assert out[0].filter_flags["bonuses"]["category"] == 0.05


def test_no_category_bonus_when_unrelated():
    j = _job(job_id="x", category="Mobile")
    out = filter_and_score([(j, 0.5)], _profile())
    assert "category" not in out[0].filter_flags


def test_location_exact_city_bonus():
    j = _job(job_id="x", location="Seattle, WA")
    out = filter_and_score([(j, 0.5)], _profile(location_city="Seattle, WA"))
    assert out[0].filter_flags["location"] == "exact_city"
    assert out[0].filter_flags["bonuses"]["location"] == 0.05


def test_location_remote_bonus_only_when_open_to_remote():
    j = _job(job_id="x", location="Remote", work_location_type="Remote")
    out_no = filter_and_score([(j, 0.5)], _profile(open_to_remote=False))
    assert "location" not in out_no[0].filter_flags

    out_yes = filter_and_score(
        [(j, 0.5)],
        _profile(open_to_remote=True, location_city="Mumbai, India"),
    )
    assert out_yes[0].filter_flags["location"] == "remote_friendly"
    assert out_yes[0].filter_flags["bonuses"]["location"] == 0.03


def test_yoe_alignment_bonus_within_window():
    j_under = _job(job_id="a", yoe_min=8)
    j_over = _job(job_id="b", yoe_min=20)
    j_match = _job(job_id="c", yoe_min=7)
    profile = _profile(years_experience=8.0)

    out = filter_and_score([(j_under, 0.5), (j_match, 0.5)], profile)
    flags_by_id = {c.job.job_id: c.filter_flags for c in out}
    assert flags_by_id["c"]["yoe"] == "yoe_aligned"
    assert flags_by_id["c"]["bonuses"]["yoe"] == 0.05

    out2 = filter_and_score([(j_over, 0.99)], profile)
    assert out2 == []  # over-qual cutoff at +2


def test_results_sorted_by_final_score_desc():
    a = _job(job_id="a", category="Mobile")
    b = _job(job_id="b", category="Backend")
    out = filter_and_score([(a, 0.55), (b, 0.50)], _profile())
    assert out[0].job.job_id == "b"  # b's primary-category bonus tips it above a
