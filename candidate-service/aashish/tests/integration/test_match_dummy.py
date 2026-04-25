"""Smoke test: /match returns the expected envelope across approaches.

P0 named this 'dummy' but the real BM25 retriever now powers it. We retain
the test name and assertions to keep the contract pinned: 10 results, full
metadata, score bounds, validation behaviour.
"""

from __future__ import annotations

SAMPLE_RESUME = (
    "DAVID WILSON\n"
    "Senior Backend Engineer with 8 years of experience. "
    "Expert in Go, Python, Postgres, Kafka, Kubernetes, AWS, microservices, gRPC."
)


def test_match_returns_up_to_ten_results(app_client):
    r = app_client.post("/match", json={"resume": {"content": SAMPLE_RESUME}})
    assert r.status_code == 200, r.text
    body = r.json()

    assert "matches" in body and isinstance(body["matches"], list)
    assert 1 <= len(body["matches"]) <= 10

    first = body["matches"][0]
    expected_keys = {
        "job_id",
        "title",
        "company",
        "match_score",
        "explanation",
        "matching_skills",
        "experience_alignment",
        "location",
        "salary_range",
        "job_category",
        "responsibilities",
        "requirements",
        "retrieval_score",
        "filter_flags",
    }
    assert expected_keys.issubset(first.keys())
    assert 0 <= first["match_score"] <= 100
    assert first["retrieval_score"] is not None and first["retrieval_score"] > 0

    meta = body["metadata"]
    expected_meta_keys = {
        "retrieval_method",
        "reranking_method",
        "processing_time_ms",
        "retrieval_count",
        "filtered_count",
        "returned_count",
        "approach",
        "cost_usd",
        "tokens",
    }
    assert expected_meta_keys.issubset(meta.keys())
    assert meta["approach"] == "bm25"
    assert meta["retrieval_method"] == "bm25"
    assert meta["reranking_method"] == "none"
    assert meta["tokens"] == {"prompt": 0, "completion": 0, "embedding": 0}


def test_match_rejects_empty_resume(app_client):
    r = app_client.post("/match", json={"resume": {"content": "  "}})
    assert r.status_code == 400


def test_match_accepts_explicit_approach(app_client):
    r = app_client.post(
        "/match",
        json={"resume": {"content": SAMPLE_RESUME}, "approach": "bm25"},
    )
    assert r.status_code == 200
    assert r.json()["metadata"]["approach"] == "bm25"


def test_match_rejects_invalid_approach(app_client):
    r = app_client.post(
        "/match",
        json={"resume": {"content": SAMPLE_RESUME}, "approach": "magic"},
    )
    assert r.status_code == 422
