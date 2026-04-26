"""Smoke test: lifespan boots, ingest seeds jobs, /health returns ok."""

from __future__ import annotations

import pytest


@pytest.mark.usefixtures("app_client")
class TestHealth:
    def test_health_returns_ok(self, app_client):
        r = app_client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["jobs_in_db"] >= 1
        assert body["openai_key_present"] is False
        assert body["default_approach"] == "bm25"
