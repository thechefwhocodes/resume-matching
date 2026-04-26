"""Extractor eval suite.

For the 5 hand-labeled resumes we compute exact-match agreement on the
categorical / boolean fields and a tolerance band on `years_experience`.
For the remaining resumes (i.e. all sample-resumes/* not in the hand-labeled
set) we run the extractor and ask the LLM to judge plausibility on a 0-5
scale (LLM-as-judge).

The full LLM-as-judge pass requires `OPENAI_API_KEY`. Without a key we report
only the field-level metrics on whatever profiles can be loaded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from eval._common import (
    RESUMES_DIR,
    load_handlabeled_profiles,
    load_resume_text,
)
from src.extractor.extractor import extract_resume_profile
from src.llm.client import CostTracker, LLMClient

log = logging.getLogger(__name__)


CATEGORICAL_FIELDS = (
    "primary_category",
    "seniority",
    "needs_h1b_sponsorship",
    "open_to_remote",
)


@dataclass
class FieldAgreement:
    field: str
    agree: int = 0
    total: int = 0

    def rate(self) -> float:
        return self.agree / self.total if self.total else float("nan")


@dataclass
class ExtractorReport:
    handlabeled_count: int = 0
    field_metrics: dict[str, float] = field(default_factory=dict)
    yoe_within_1_yr: float = float("nan")
    secondary_jaccard_mean: float = float("nan")
    skills_jaccard_mean: float = float("nan")
    judged_count: int = 0
    judge_mean: float = float("nan")
    notes: list[str] = field(default_factory=list)


def _jaccard(a: list[str], b: list[str]) -> float:
    set_a = {x.lower() for x in a or []}
    set_b = {x.lower() for x in b or []}
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def run_extractor_eval(
    *,
    session: Session,
    llm: LLMClient,
    judge_others: bool = True,
) -> ExtractorReport:
    report = ExtractorReport()
    handlabeled = load_handlabeled_profiles()
    report.handlabeled_count = len(handlabeled)

    if not handlabeled:
        report.notes.append("No hand-labeled profiles found.")
        return report

    if not llm.is_available:
        report.notes.append("OPENAI_API_KEY missing -- extractor eval skipped.")
        return report

    tracker = CostTracker()

    fields = {f: FieldAgreement(field=f) for f in CATEGORICAL_FIELDS}
    yoe_close = 0
    secondary_jaccs: list[float] = []
    skills_jaccs: list[float] = []

    for resume_file, gold in handlabeled.items():
        try:
            text = load_resume_text(resume_file)
        except FileNotFoundError as e:
            report.notes.append(str(e))
            continue
        predicted = extract_resume_profile(
            session,
            content=text,
            llm=llm,
            tracker=tracker,
            filename=resume_file,
        ).model_dump()

        for f in CATEGORICAL_FIELDS:
            fields[f].total += 1
            if predicted.get(f) == gold.get(f):
                fields[f].agree += 1

        gold_yoe = float(gold.get("years_experience", 0))
        pred_yoe = float(predicted.get("years_experience", 0))
        if abs(gold_yoe - pred_yoe) <= 1.0:
            yoe_close += 1

        secondary_jaccs.append(
            _jaccard(gold.get("secondary_categories", []), predicted.get("secondary_categories", []))
        )
        skills_jaccs.append(_jaccard(gold.get("skills", []), predicted.get("skills", [])))

    report.field_metrics = {f: fa.rate() for f, fa in fields.items()}
    report.yoe_within_1_yr = yoe_close / max(1, len(handlabeled))
    report.secondary_jaccard_mean = sum(secondary_jaccs) / len(secondary_jaccs)
    report.skills_jaccard_mean = sum(skills_jaccs) / len(skills_jaccs)

    if judge_others and llm.is_available:
        judged: list[float] = []
        for path in sorted(RESUMES_DIR.rglob("*.txt")):
            rel = str(path.relative_to(RESUMES_DIR))
            if rel in handlabeled:
                continue
            try:
                text = load_resume_text(rel)
            except FileNotFoundError:
                continue
            profile = extract_resume_profile(
                session, content=text, llm=llm, tracker=tracker, filename=rel
            )
            score = _llm_judge(text, profile.model_dump(), llm=llm, tracker=tracker)
            if score is not None:
                judged.append(score)
        if judged:
            report.judged_count = len(judged)
            report.judge_mean = sum(judged) / len(judged)

    report.notes.append(
        f"cost: ${tracker.cost_usd:.4f} (prompt={tracker.prompt} completion={tracker.completion} embed={tracker.embedding})"
    )
    return report


def _llm_judge(
    resume_text: str,
    profile: dict[str, Any],
    *,
    llm: LLMClient,
    tracker: CostTracker,
) -> float | None:
    """Ask the model to rate the extracted profile 0-5 for fidelity."""
    from pydantic import BaseModel, Field  # noqa: PLC0415

    class JudgeOutput(BaseModel):
        score: float = Field(ge=0, le=5)
        rationale: str = Field(default="", max_length=200)

    sys_prompt = (
        "You are a strict reviewer grading whether an extracted profile faithfully "
        "represents a resume. Score 0-5: 0=hallucinated, 5=accurate and complete. "
        "Be calibrated; most automated extractions land in the 3-4 range."
    )
    user_prompt = (
        "RESUME (truncated):\n"
        f"{resume_text[:3000]}\n\n"
        "EXTRACTED PROFILE (JSON):\n"
        f"{profile}\n"
    )
    try:
        out = llm.chat_structured(
            response_model=JudgeOutput,
            system=sys_prompt,
            user=user_prompt,
            model=llm.settings.extract_model,
            temperature=0.0,
            seed=42,
            tracker=tracker,
        )
        return float(out.score)
    except Exception as e:
        log.warning("extractor judge failed: %s", e)
        return None
