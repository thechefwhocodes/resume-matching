"""Eval orchestrator. Runs all 3 suites and writes a timestamped markdown report.

Usage:
    make eval                        # validates GT, then runs this
    python -m eval.report            # equivalent
    python -m eval.report --suite extractor

Each invocation also writes one EvalRun row per suite into Postgres for
historical tracking.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from dataclasses import asdict
from typing import Any

from eval._common import RESULTS_DIR
from eval.extractor_eval import run_extractor_eval
from eval.reranker_eval import run_reranker_eval
from eval.retriever_eval import run_retriever_eval
from src.db.engine import SessionLocal, get_engine
from src.db.models import Base, EvalRun
from src.llm.client import get_llm_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
log = logging.getLogger("aashish.eval.report")


def _ensure_db():
    eng = get_engine()
    Base.metadata.create_all(bind=eng)


def _persist_eval_run(suite: str, payload: dict[str, Any], notes: str | None = None) -> None:
    with SessionLocal() as session:
        session.add(EvalRun(suite=suite, metrics=payload, notes=notes))
        session.commit()


def _format_extractor_md(report) -> str:
    lines = [
        "## Extractor",
        "",
        f"- hand-labeled resumes scored: **{report.handlabeled_count}**",
        f"- LLM-judged resumes (others): **{report.judged_count}**",
        "",
        "### Field-level agreement (hand-labeled)",
        "",
        "| Field | Agreement |",
        "| --- | --- |",
    ]
    for f, rate in report.field_metrics.items():
        lines.append(f"| `{f}` | {_fmt_rate(rate)} |")
    lines.extend([
        f"| years_experience (within 1 yr) | {_fmt_rate(report.yoe_within_1_yr)} |",
        f"| secondary_categories Jaccard | {_fmt_rate(report.secondary_jaccard_mean)} |",
        f"| skills Jaccard | {_fmt_rate(report.skills_jaccard_mean)} |",
        "",
        f"### LLM-judge mean (others, 0-5): **{_fmt_score(report.judge_mean)}**",
    ])
    if report.notes:
        lines.append("")
        lines.append("### Notes")
        for n in report.notes:
            lines.append(f"- {n}")
    return "\n".join(lines)


def _format_retriever_md(report) -> str:
    lines = ["## Retriever", ""]
    if not report.approaches:
        lines.append("_No populated ground-truth picks; skipped._")
    else:
        lines.append("| Approach | n | R@5 | R@10 | R@30 | R@50 | MRR | Coverage |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for a in report.approaches:
            r = a["recall"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        a["name"],
                        str(a["n_queries"]),
                        _fmt_rate(r["@5"]),
                        _fmt_rate(r["@10"]),
                        _fmt_rate(r["@30"]),
                        _fmt_rate(r["@50"]),
                        _fmt_rate(a["mrr"]),
                        _fmt_rate(a["coverage"]),
                    ]
                )
                + " |"
            )
        if report.rrf_hybrid_added:
            lines.append("")
            lines.append("> **R@30 gap small enough that RRF-hybrid is worth shipping.**")
    if report.notes:
        lines.append("")
        lines.append("### Notes")
        for n in report.notes:
            lines.append(f"- {n}")
    return "\n".join(lines)


def _format_reranker_md(report) -> str:
    lines = [
        "## Reranker",
        "",
        f"- queries: **{report.n_queries}**",
        f"- NDCG@10: **{_fmt_rate(report.ndcg_at_10)}**",
        f"- MRR@10: **{_fmt_rate(report.mrr_at_10)}**",
        f"- Precision@5: **{_fmt_rate(report.precision_at_5)}**",
        f"- score mean / std: **{_fmt_score(report.score_mean)} ± {_fmt_score(report.score_std)}**",
        f"- Cohen's kappa vs judge: **{_fmt_score(report.kappa_vs_judge)}**",
    ]
    if report.notes:
        lines.append("")
        lines.append("### Notes")
        for n in report.notes:
            lines.append(f"- {n}")
    return "\n".join(lines)


def _fmt_rate(x: float) -> str:
    if x != x:  # NaN
        return "_n/a_"
    return f"{x:.3f}"


def _fmt_score(x: float) -> str:
    if x != x:
        return "_n/a_"
    return f"{x:.2f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        choices=("extractor", "retriever", "reranker", "all"),
        default="all",
    )
    args = parser.parse_args()

    _ensure_db()

    llm = get_llm_client()
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = RESULTS_DIR / f"{timestamp}.md"
    sections: list[str] = [
        "# Resume Matching Eval Report",
        "",
        f"- timestamp: `{timestamp}`",
        f"- openai_key_present: `{llm.is_available}`",
        f"- embed_model: `{llm.settings.embed_model}`",
        f"- extract_model: `{llm.settings.extract_model}`",
        f"- rerank_model: `{llm.settings.rerank_model}`",
        "",
    ]

    payloads: dict[str, dict[str, Any]] = {}

    with SessionLocal() as session:
        if args.suite in ("extractor", "all"):
            log.info("running extractor eval...")
            r = run_extractor_eval(session=session, llm=llm)
            sections.append(_format_extractor_md(r))
            sections.append("")
            payloads["extractor"] = _to_dict(r)

        if args.suite in ("retriever", "all"):
            log.info("running retriever eval...")
            r = run_retriever_eval(session=session, llm=llm)
            sections.append(_format_retriever_md(r))
            sections.append("")
            payloads["retriever"] = _to_dict(r)

        if args.suite in ("reranker", "all"):
            log.info("running reranker eval...")
            r = run_reranker_eval(session=session, llm=llm)
            sections.append(_format_reranker_md(r))
            sections.append("")
            payloads["reranker"] = _to_dict(r)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(sections))
    log.info("wrote %s", out_path)

    for suite, payload in payloads.items():
        try:
            _persist_eval_run(suite=suite, payload=payload)
        except Exception as e:
            log.warning("failed to persist EvalRun for %s: %s", suite, e)

    print(f"\nReport written: {out_path}")
    return 0


def _to_dict(obj) -> dict:
    try:
        d = asdict(obj)
    except TypeError:
        d = json.loads(json.dumps(obj, default=str))
    return _scrub_nans(d)


def _scrub_nans(value):
    """Recursively replace NaN floats with None so Postgres JSON accepts them."""
    if isinstance(value, float):
        if value != value:  # NaN check
            return None
        return value
    if isinstance(value, dict):
        return {k: _scrub_nans(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub_nans(v) for v in value]
    if isinstance(value, tuple):
        return [_scrub_nans(v) for v in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
