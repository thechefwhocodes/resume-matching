"""Common retriever protocol shared by BM25 and embedding retrievers.

Each retriever takes the resume text (and optional metadata) and returns a
ranked list of `(job_id, retrieval_score)` tuples. The pipeline takes care of
turning these into JobMatch objects + downstream rerank.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Retriever(Protocol):
    name: str

    def retrieve(self, query: str, *, top_k: int) -> list[tuple[str, float]]:
        """Return up to top_k (job_id, score) pairs sorted by descending score."""
        ...
