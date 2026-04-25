"""Retriever eval suite.

Compares 3 retrieval approaches on the locked ground-truth picks:
    * bm25
    * embed
    * embed+filters    (post-filter on top of embed)

For each approach we report Recall@30, Recall@50, MRR over the top-50, and
candidate coverage (how many GT items were retrieved at all).

Per the plan, an `rrf-hybrid` is added only when bm25 vs embed Recall@30 are
within ~5pts. The orchestrator (report.py) decides this conditionally.

When the GT picks file is empty for a resume, that resume is skipped.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from eval._common import (
    GroundTruthPick,
    load_picks,
    load_resume_text,
    mrr,
    recall_at_k,
)
from src.db.queries import retrieve_with_filters
from src.extractor.extractor import extract_resume_profile
from src.extractor.prompts import ResumeProfile
from src.llm.client import CostTracker, LLMClient
from src.retrieval.base import Retriever
from src.retrieval.bm25 import BM25Index
from src.retrieval.embeddings import EmbeddingRetriever

log = logging.getLogger(__name__)

K_VALUES = (5, 10, 30, 50)


@dataclass
class ApproachMetrics:
    name: str
    n_queries: int = 0
    recall: dict[int, list[float]] = field(default_factory=dict)
    mrr_scores: list[float] = field(default_factory=list)
    coverage: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "name": self.name,
            "n_queries": self.n_queries,
            "recall": {
                f"@{k}": _mean(self.recall.get(k, [])) for k in K_VALUES
            },
            "mrr": _mean(self.mrr_scores),
            "coverage": _mean(self.coverage),
        }


def _mean(xs: Iterable[float]) -> float:
    xs = [x for x in xs if x == x]  # drop NaN
    return sum(xs) / len(xs) if xs else float("nan")


@dataclass
class RetrieverReport:
    approaches: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    rrf_hybrid_added: bool = False


def _retrieve_bm25(bm25_index: Retriever, query: str, k: int) -> list[str]:
    return [jid for jid, _ in bm25_index.retrieve(query, top_k=k)]


def _retrieve_embed(
    *, llm: LLMClient, session: Session, tracker: CostTracker, query: str, k: int
) -> list[str]:
    if not llm.is_available:
        return []
    retr = EmbeddingRetriever(llm=llm, session=session, tracker=tracker)
    return [jid for jid, _ in retr.retrieve(query, top_k=k)]


def _retrieve_embed_filters(
    *,
    llm: LLMClient,
    session: Session,
    tracker: CostTracker,
    query: str,
    profile: ResumeProfile,
    k: int,
) -> list[str]:
    if not llm.is_available:
        return []
    retr = EmbeddingRetriever(llm=llm, session=session, tracker=tracker)
    pairs = retr.retrieve(query, top_k=k)
    survivors = retrieve_with_filters(
        session, candidate_pairs=pairs, profile=profile
    )
    return [c.job.job_id for c in survivors][:k]


def run_retriever_eval(
    *,
    session: Session,
    llm: LLMClient,
) -> RetrieverReport:
    report = RetrieverReport()

    picks = [p for p in load_picks() if p.expected_job_ids]
    if not picks:
        report.notes.append(
            "No populated ground-truth picks (all empty placeholders). Retriever metrics skipped."
        )
        return report

    bm25_index = BM25Index.build_from_session(session)
    metrics_bm25 = ApproachMetrics("bm25")
    metrics_embed = ApproachMetrics("embed")
    metrics_filt = ApproachMetrics("embed+filters")
    tracker = CostTracker()

    for pick in picks:
        try:
            text = load_resume_text(pick.resume_file)
        except FileNotFoundError as e:
            report.notes.append(str(e))
            continue
        gt = set(pick.expected_job_ids)

        # bm25
        retrieved = _retrieve_bm25(bm25_index, text, k=max(K_VALUES))
        _accumulate(metrics_bm25, retrieved, gt)

        if llm.is_available:
            # embed
            retrieved = _retrieve_embed(
                llm=llm, session=session, tracker=tracker, query=text, k=max(K_VALUES)
            )
            _accumulate(metrics_embed, retrieved, gt)

            # embed+filters
            profile = extract_resume_profile(
                session, content=text, llm=llm, tracker=tracker, filename=pick.resume_file
            )
            retrieved = _retrieve_embed_filters(
                llm=llm,
                session=session,
                tracker=tracker,
                query=text,
                profile=profile,
                k=max(K_VALUES),
            )
            _accumulate(metrics_filt, retrieved, gt)

    summaries = [metrics_bm25.summary()]
    if llm.is_available:
        summaries.extend([metrics_embed.summary(), metrics_filt.summary()])
        # RRF hybrid only if bm25 vs embed within ~5pts at @30
        bm25_r30 = summaries[0]["recall"]["@30"]
        embed_r30 = summaries[1]["recall"]["@30"]
        if bm25_r30 == bm25_r30 and embed_r30 == embed_r30:
            if abs(bm25_r30 - embed_r30) <= 0.05:
                report.notes.append(
                    f"recall@30 gap ({abs(bm25_r30 - embed_r30):.3f}) within 0.05; "
                    "RRF-hybrid would be worth adding."
                )
                report.rrf_hybrid_added = True
            else:
                report.notes.append(
                    f"recall@30 gap ({abs(bm25_r30 - embed_r30):.3f}) > 0.05; "
                    "skipping RRF-hybrid (per plan)."
                )

    report.approaches = summaries
    report.notes.append(
        f"cost: ${tracker.cost_usd:.4f} (prompt={tracker.prompt} completion={tracker.completion} embed={tracker.embedding})"
    )
    return report


def _accumulate(metrics: ApproachMetrics, retrieved: list[str], gt: set[str]) -> None:
    metrics.n_queries += 1
    for k in K_VALUES:
        metrics.recall.setdefault(k, []).append(recall_at_k(retrieved, gt, k))
    metrics.mrr_scores.append(mrr(retrieved, gt))
    coverage = len([j for j in retrieved if j in gt]) / max(1, len(gt))
    metrics.coverage.append(coverage)


def _picks_for_test() -> list[GroundTruthPick]:
    return load_picks()
