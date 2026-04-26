"""Verify the pgvector cosine query path using a fake LLMClient.

The retriever doesn't care whether vectors come from OpenAI or a stub --
it just needs `embed()` to return a vector of the right dim. We seed a few
jobs with known vectors and confirm cosine-distance ordering works end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest


@dataclass
class _FakeSettings:
    embedding_dim: int = 1536
    embed_model: str = "fake-embed"


class _FakeLLM:
    def __init__(self):
        self.settings = _FakeSettings()
        self.is_available = True

    def embed(self, texts, *, tracker=None):
        out = []
        for t in texts:
            v = [0.0] * self.settings.embedding_dim
            v[0] = 1.0 if "alpha" in t.lower() else 0.0
            v[1] = 1.0 if "beta" in t.lower() else 0.0
            v[2] = 1.0 if "gamma" in t.lower() else 0.0
            out.append(v)
        return out


@pytest.mark.usefixtures("configured_env")
def test_embedding_retriever_orders_by_cosine_similarity():
    from src.db.engine import SessionLocal, get_engine
    from src.db.models import Base, Job
    from src.llm.client import CostTracker
    from src.retrieval.embeddings import EmbeddingRetriever

    eng = get_engine()
    Base.metadata.drop_all(bind=eng)
    Base.metadata.create_all(bind=eng)

    fake_llm = _FakeLLM()

    def vec(idx: int) -> list[float]:
        v = [0.0] * fake_llm.settings.embedding_dim
        v[idx] = 1.0
        return v

    with SessionLocal() as session:
        session.add_all([
            Job(
                job_id="job-alpha",
                title="Alpha",
                company_name="A",
                content_sha256="a" * 64,
                embedding=vec(0),
                status="active",
                h1b_sponsorship=False,
            ),
            Job(
                job_id="job-beta",
                title="Beta",
                company_name="B",
                content_sha256="b" * 64,
                embedding=vec(1),
                status="active",
                h1b_sponsorship=False,
            ),
            Job(
                job_id="job-gamma",
                title="Gamma",
                company_name="G",
                content_sha256="c" * 64,
                embedding=vec(2),
                status="active",
                h1b_sponsorship=False,
            ),
        ])
        session.commit()

        retr = EmbeddingRetriever(
            llm=fake_llm,  # type: ignore[arg-type]
            session=session,
            tracker=CostTracker(),
        )

        result = retr.retrieve("an alpha document", top_k=3)
        assert result[0][0] == "job-alpha"
        assert result[0][1] == pytest.approx(1.0, abs=1e-3)

        result = retr.retrieve("a beta document", top_k=2)
        assert result[0][0] == "job-beta"
