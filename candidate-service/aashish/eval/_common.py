"""Shared helpers used by extractor / retriever / reranker eval suites.

Centralised so the report can import metrics and so each suite can be invoked
standalone for fast iteration.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GT_DIR = Path(__file__).resolve().parent / "ground_truth"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESUMES_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "sample-resumes"


@dataclass(slots=True)
class GroundTruthPick:
    resume_file: str
    expected_job_ids: list[str]
    notes: str = ""


def load_picks() -> list[GroundTruthPick]:
    out: list[GroundTruthPick] = []
    for p in sorted(GT_DIR.glob("*.json")):
        if p.name == "profiles_handlabeled.json":
            continue
        data = json.loads(p.read_text())
        out.append(
            GroundTruthPick(
                resume_file=data.get("resume_file", p.stem),
                expected_job_ids=list(data.get("expected_job_ids") or []),
                notes=data.get("notes", ""),
            )
        )
    return out


def load_handlabeled_profiles() -> dict[str, dict[str, Any]]:
    fp = GT_DIR / "profiles_handlabeled.json"
    if not fp.exists():
        return {}
    rows = json.loads(fp.read_text())
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        if "resume_file" in r and "profile" in r:
            out[r["resume_file"]] = r["profile"]
    return out


def load_resume_text(resume_file: str) -> str:
    fp = RESUMES_DIR / resume_file
    if not fp.exists():
        raise FileNotFoundError(f"resume not found: {fp}")
    return fp.read_text()


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return float("nan")
    top = retrieved[:k]
    hits = sum(1 for jid in top if jid in relevant)
    return hits / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not retrieved:
        return 0.0
    top = retrieved[:k]
    if not top:
        return 0.0
    hits = sum(1 for jid in top if jid in relevant)
    return hits / len(top)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    for i, jid in enumerate(retrieved):
        if jid in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant_ranked: list[str], k: int) -> float:
    """NDCG with binary relevance graded by ideal position.

    `relevant_ranked` is the GT order (best first). We score relevance as
    `len(relevant_ranked) - idx` so position-1 is most valuable.
    """
    if not relevant_ranked:
        return float("nan")
    grades = {jid: len(relevant_ranked) - i for i, jid in enumerate(relevant_ranked)}

    def _dcg(items: list[str]) -> float:
        return sum(
            (grades.get(jid, 0)) / math.log2(i + 2) for i, jid in enumerate(items[:k])
        )

    actual = _dcg(retrieved[:k])
    ideal = _dcg(relevant_ranked[:k])
    return actual / ideal if ideal > 0 else 0.0


def cohens_kappa(rater_a: list[float], rater_b: list[float], *, buckets: int = 5) -> float:
    """Quadratic-ish Cohen's kappa for ordinal ratings.

    Buckets the floats into `buckets` equal-width bins over [0, 100], then
    computes standard kappa between the two raters. Returns NaN if either
    rater is constant.
    """
    if len(rater_a) != len(rater_b) or not rater_a:
        return float("nan")

    def _bucket(x: float) -> int:
        b = int(x // (100 / buckets))
        return min(buckets - 1, max(0, b))

    a = [_bucket(x) for x in rater_a]
    b = [_bucket(x) for x in rater_b]
    n = len(a)
    if len(set(a)) == 1 or len(set(b)) == 1:
        return float("nan")

    agree = sum(1 for x, y in zip(a, b, strict=False) if x == y) / n
    counts_a = [0] * buckets
    counts_b = [0] * buckets
    for x in a:
        counts_a[x] += 1
    for y in b:
        counts_b[y] += 1
    chance = sum((counts_a[i] / n) * (counts_b[i] / n) for i in range(buckets))
    if chance == 1.0:
        return float("nan")
    return (agree - chance) / (1 - chance)
