"""Dense retriever: embed the resume, then query Postgres via pgvector cosine.

The Job table stores `embedding Vector(1536)` populated at ingest. Query path:
    1. Embed the resume text (single API call, ~100-200ms).
    2. SELECT job_id, 1 - (embedding <=> :q) AS sim ORDER BY embedding <=> :q LIMIT N
       (we use the SQLAlchemy ORM's `cosine_distance` helper from pgvector).
    3. Return (job_id, similarity) pairs sorted by descending similarity.

Falls back gracefully when the LLM client is unavailable (zero-vector query
won't match anything meaningful, so we return [] and let the pipeline fail
back to BM25 with a warning).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.models import Job
from src.llm.client import CostTracker, LLMClient

log = logging.getLogger(__name__)


@dataclass(slots=True)
class EmbeddingRetriever:
    """Wrap an LLMClient + a Session-bound query.

    A new instance is created per request because it carries the per-request
    CostTracker. The actual retrieval is one embedding call + one SQL query.
    """

    llm: LLMClient
    session: Session
    tracker: CostTracker
    name: str = "embed"

    def retrieve(self, query: str, *, top_k: int) -> list[tuple[str, float]]:
        if not self.llm.is_available:
            log.warning("embed retriever called without OPENAI_API_KEY; returning []")
            return []

        vectors = self.llm.embed([query], tracker=self.tracker)
        if not vectors:
            return []
        q_vec = vectors[0]

        distance = Job.embedding.cosine_distance(q_vec)
        stmt = (
            select(Job.job_id, distance.label("dist"))
            .where(Job.status == "active")
            .order_by(distance)
            .limit(top_k)
        )
        rows = self.session.execute(stmt).all()
        return [(jid, float(1.0 - dist)) for jid, dist in rows]
