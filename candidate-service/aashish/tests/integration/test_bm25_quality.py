"""Sanity check: BM25 surfaces topically relevant jobs for a strong resume.

We don't assert exact rankings (BM25 is brittle to exact wording) -- we just
ensure the top 10 contains at least one Backend or ML/AI category job for a
backend engineer resume, and that no result has score 0.
"""

from __future__ import annotations

BACKEND_RESUME = (
    "Senior Backend Engineer with 8 years experience in Go, Python, gRPC, "
    "Postgres, Kafka, Kubernetes, AWS. Built microservices at Snowflake "
    "and Expedia handling millions of requests per second."
)


def test_bm25_returns_relevant_categories(app_client):
    r = app_client.post(
        "/match",
        json={"resume": {"content": BACKEND_RESUME}, "approach": "bm25"},
    )
    assert r.status_code == 200
    matches = r.json()["matches"]
    assert len(matches) > 0

    categories = {m.get("job_category") for m in matches}
    assert categories & {"Backend", "Full-Stack", "ML/AI", "DevOps", "Data"}, (
        f"expected backend-adjacent categories in top results, got {categories}"
    )

    for m in matches:
        assert m["retrieval_score"] is not None and m["retrieval_score"] > 0


def test_bm25_empty_query_safe(app_client):
    r = app_client.post(
        "/match",
        json={"resume": {"content": "................"}, "approach": "bm25"},
    )
    assert r.status_code == 200
