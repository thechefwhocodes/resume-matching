"""Prompts + Pydantic schema for the resume profile extractor.

The extractor produces a compact `ResumeProfile` we use for SQL hard-filters
(H1B sponsorship needed, severe YoE under-qualification) and soft-scoring
(category alignment, location bonus). Any change here invalidates the
`resume_profile_cache` table -- bump the model version string when needed.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PROFILE_PROMPT_VERSION = "v1"

JOB_CATEGORIES = (
    "Backend",
    "Frontend",
    "Full-Stack",
    "ML/AI",
    "Data",
    "DevOps",
    "Mobile",
    "Other",
)

SeniorityLevel = Literal["Junior", "Mid", "Senior", "Staff+"]
JobCategory = Literal[
    "Backend", "Frontend", "Full-Stack", "ML/AI", "Data", "DevOps", "Mobile", "Other"
]


class ResumeProfile(BaseModel):
    """Structured view of a resume used by filters + scoring + display."""

    model_config = ConfigDict(extra="ignore")

    years_experience: float = Field(
        ge=0,
        le=40,
        description=(
            "Total full-time years of professional software engineering "
            "experience. Internships count at 0.25 each. Round to one decimal."
        ),
    )
    primary_category: JobCategory = Field(
        description=(
            "The single category that best describes the candidate's last 3-5 "
            "years of work. Choose 'Other' only if none fits."
        )
    )
    secondary_categories: list[JobCategory] = Field(
        default_factory=list,
        description=(
            "Up to 2 other categories the candidate has clear experience in. "
            "Empty list if specialised."
        ),
        max_length=2,
    )
    seniority: SeniorityLevel = Field(
        description=(
            "Junior=0-2y, Mid=3-5y, Senior=6-8y, Staff+=9+y. Use scope/leadership "
            "as a tiebreaker, not just years."
        )
    )
    needs_h1b_sponsorship: bool = Field(
        description=(
            "True if any signal suggests the candidate is on or will need a US "
            "work visa (e.g. 'currently on F-1 OPT', 'requires sponsorship', "
            "'Indian citizen, looking for US roles'). Default False if unstated."
        )
    )
    open_to_remote: bool = Field(
        description=(
            "True if the resume mentions remote work preference, current "
            "remote role, or location is outside the US."
        )
    )
    location_city: str | None = Field(
        default=None,
        description=(
            "Most recent or stated home city (e.g. 'Seattle, WA', 'Bangalore, India'). "
            "Null if unstated."
        ),
    )
    skills: list[str] = Field(
        default_factory=list,
        max_length=30,
        description=(
            "Up to 30 normalised technical skills (Go, Python, Postgres, etc). "
            "Lowercase product names, normalise common variants ('postgres' not 'PostgreSQL')."
        ),
    )
    summary: str = Field(
        default="",
        max_length=400,
        description="One-sentence summary of who this candidate is.",
    )


SYSTEM_PROMPT = (
    "You extract structured engineering profiles from resumes. "
    "Be concise and conservative. Never invent skills or experience that the "
    "resume does not clearly state. Return only the structured object."
)


def build_user_prompt(resume_text: str) -> str:
    return (
        "Resume below. Extract a ResumeProfile for matching against engineering "
        "job descriptions.\n\n"
        "----- RESUME START -----\n"
        f"{resume_text.strip()}\n"
        "----- RESUME END -----\n"
    )
