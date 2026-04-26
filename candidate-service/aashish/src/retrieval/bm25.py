"""In-memory BM25 retriever, built once at startup from the jobs table.

Tokenisation is intentionally simple: lowercase + alphanumeric runs. We keep
plain-text representations of `responsibilities` (already HTML-stripped during
ingest) and `requirements` plus the `title` so the index sees the same content
the embedder does.

The index is built off the SQLAlchemy session; the constructor is the single
hot path. After that, lookups are O(N * tokens) which for N=300 and tokens<200
is well under 5ms in practice.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi
from sqlalchemy.orm import Session

from src.db.models import Job
from src.db.queries import list_active_jobs

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9_+#./-]+")


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    return [tok.lower() for tok in _TOKEN_RE.findall(text)]


def _document_for(job: Job) -> str:
    parts = [
        job.title or "",
        job.job_category or "",
        job.responsibilities_text or "",
        job.requirements or "",
    ]
    return "\n".join(p for p in parts if p)


@dataclass(slots=True)
class BM25Index:
    job_ids: list[str]
    bm25: BM25Okapi
    name: str = "bm25"

    @classmethod
    def build_from_session(cls, session: Session) -> BM25Index:
        jobs = list_active_jobs(session)
        ids = [j.job_id for j in jobs]
        corpus = [_tokenize(_document_for(j)) for j in jobs]
        if not corpus:
            log.warning("bm25: no jobs found at index-build time")
            return cls(job_ids=[], bm25=BM25Okapi([[""]]))
        log.info(
            "bm25: built index over %d jobs, mean tokens/doc=%.1f",
            len(corpus),
            sum(len(d) for d in corpus) / len(corpus),
        )
        return cls(job_ids=ids, bm25=BM25Okapi(corpus))

    def retrieve(self, query: str, *, top_k: int) -> list[tuple[str, float]]:
        tokens = _tokenize(query)
        if not tokens or not self.job_ids:
            return []
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(
            zip(self.job_ids, scores, strict=True),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(jid, float(score)) for jid, score in ranked[:top_k] if score > 0]
