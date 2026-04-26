"""Reranker eval suite.

Runs the rerank step (reads same candidates the retriever produces) and
computes NDCG@10, MRR@10, Precision@5, plus a calibration check (mean / std
of returned fit_score) and Cohen's kappa against a second model judge.

Includes a tiny on-disk cache keyed by `hash(resume + ranked_job_ids + model)`
so re-running `make eval` doesn't re-pay for tokens.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy.orm import Session

from eval._common import (
    cohens_kappa,
    load_picks,
    load_resume_text,
    mrr,
    ndcg_at_k,
    precision_at_k,
)
from src.db.models import Job
from src.db.queries import retrieve_with_filters
from src.extractor.extractor import extract_resume_profile
from src.llm.client import CostTracker, LLMClient
from src.rerank.reranker import rerank
from src.retrieval.embeddings import EmbeddingRetriever

log = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)


@dataclass
class RerankerReport:
    n_queries: int = 0
    ndcg_at_10: float = float("nan")
    mrr_at_10: float = float("nan")
    precision_at_5: float = float("nan")
    score_mean: float = float("nan")
    score_std: float = float("nan")
    kappa_vs_judge: float = float("nan")
    notes: list[str] = field(default_factory=list)


def _cache_key(*, resume: str, ids: list[str], model: str) -> str:
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(b"|")
    h.update(resume.encode())
    h.update(b"|")
    h.update("\n".join(ids).encode())
    return h.hexdigest()[:24]


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"rerank-{key}.json"


def aggregate_rater_scores(
    *,
    ids: list[str],
    primary: dict[str, dict],
    judge: dict[str, dict] | None,
) -> tuple[list[float], list[float]]:
    """Build aligned score lists for calibration + cohen's kappa.

    When `judge` is None (single-model mode) the primary list contains every
    primary score and the judge list is empty.  When `judge` is provided we
    only emit rows where BOTH raters scored the same id, guaranteeing
    `len(primary_scores) == len(judge_scores)` so `cohens_kappa` does not hit
    its alignment guard.
    """
    primary_scores: list[float] = []
    judge_scores: list[float] = []
    if judge is None:
        primary_scores.extend(
            primary[jid]["fit_score"] for jid in ids if jid in primary
        )
        return primary_scores, judge_scores
    for jid in ids:
        if jid in primary and jid in judge:
            primary_scores.append(primary[jid]["fit_score"])
            judge_scores.append(judge[jid]["fit_score"])
    return primary_scores, judge_scores


def _cached_or_call(
    *,
    cache_key: str,
    resume: str,
    profile,
    candidates: list[Job],
    llm: LLMClient,
    tracker: CostTracker,
):
    fp = _cache_path(cache_key)
    if fp.exists():
        log.info("rerank cache HIT key=%s", cache_key)
        return json.loads(fp.read_text())

    outcome = rerank(
        resume_text=resume,
        profile=profile,
        candidates=candidates,
        llm=llm,
        tracker=tracker,
    )
    serialised = {
        jid: {
            "fit_score": item.fit_score,
            "reasons": item.reasons,
            "concerns": item.concerns,
            "explanation": item.explanation,
        }
        for jid, item in outcome.items_by_job_id.items()
    }
    fp.write_text(json.dumps(serialised, indent=2))
    return serialised


def run_reranker_eval(
    *,
    session: Session,
    llm: LLMClient,
) -> RerankerReport:
    report = RerankerReport()
    if not llm.is_available:
        report.notes.append("OPENAI_API_KEY missing -- reranker eval skipped.")
        return report

    picks = [p for p in load_picks() if p.expected_job_ids]
    if not picks:
        report.notes.append("No populated GT picks; reranker metrics skipped.")
        return report

    tracker = CostTracker()
    ndcgs: list[float] = []
    mrrs: list[float] = []
    p5s: list[float] = []
    all_scores_primary: list[float] = []
    all_scores_judge: list[float] = []

    for pick in picks:
        try:
            text = load_resume_text(pick.resume_file)
        except FileNotFoundError as e:
            report.notes.append(str(e))
            continue

        profile = extract_resume_profile(
            session, content=text, llm=llm, tracker=tracker, filename=pick.resume_file
        )
        retr = EmbeddingRetriever(llm=llm, session=session, tracker=tracker)
        pairs = retr.retrieve(text, top_k=30)
        survivors = retrieve_with_filters(
            session, candidate_pairs=pairs, profile=profile
        )
        if not survivors:
            continue
        candidates = [c.job for c in survivors][:30]
        ids = [c.job_id for c in candidates]

        primary = _cached_or_call(
            cache_key=_cache_key(resume=text, ids=ids, model=llm.settings.rerank_model),
            resume=text,
            profile=profile,
            candidates=candidates,
            llm=llm,
            tracker=tracker,
        )
        ranked = sorted(
            primary.items(), key=lambda kv: kv[1]["fit_score"], reverse=True
        )
        ranked_ids = [jid for jid, _ in ranked]

        ndcgs.append(ndcg_at_k(ranked_ids, pick.expected_job_ids, 10))
        mrrs.append(mrr(ranked_ids[:10], set(pick.expected_job_ids)))
        p5s.append(precision_at_k(ranked_ids, set(pick.expected_job_ids), 5))

        judge: dict[str, dict] | None = None
        if llm.settings.extract_model != llm.settings.rerank_model:
            judge = _cached_or_call(
                cache_key=_cache_key(
                    resume=text, ids=ids, model=llm.settings.extract_model
                ),
                resume=text,
                profile=profile,
                candidates=candidates,
                llm=llm,
                tracker=tracker,
            )
        p_scores, j_scores = aggregate_rater_scores(
            ids=ids, primary=primary, judge=judge
        )
        all_scores_primary.extend(p_scores)
        all_scores_judge.extend(j_scores)

    report.n_queries = len(ndcgs)
    if ndcgs:
        report.ndcg_at_10 = sum(ndcgs) / len(ndcgs)
        report.mrr_at_10 = sum(mrrs) / len(mrrs)
        report.precision_at_5 = sum(p5s) / len(p5s)
    if all_scores_primary:
        n = len(all_scores_primary)
        mean = sum(all_scores_primary) / n
        var = sum((x - mean) ** 2 for x in all_scores_primary) / max(1, n - 1)
        report.score_mean = mean
        report.score_std = var**0.5
    if all_scores_judge:
        report.kappa_vs_judge = cohens_kappa(all_scores_primary, all_scores_judge)
    elif llm.settings.extract_model == llm.settings.rerank_model:
        report.notes.append(
            "EXTRACT_MODEL == RERANK_MODEL; skipping cross-model kappa "
            "(set them differently to enable inter-rater agreement)."
        )

    report.notes.append(
        f"cost: ${tracker.cost_usd:.4f} (prompt={tracker.prompt} completion={tracker.completion} embed={tracker.embedding})"
    )
    return report
