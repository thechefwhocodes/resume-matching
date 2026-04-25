"""Shared pytest fixtures.

Integration tests boot an ephemeral pgvector-enabled Postgres via testcontainers
and rebuild the schema for each test session. The container image is pinned to
match the one we ship in docker-compose.yml so behaviour stays identical.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def postgres_container() -> Iterator[str]:
    """Spin up a pgvector-enabled Postgres container; yield its DSN."""
    pytest.importorskip("testcontainers")
    from testcontainers.postgres import PostgresContainer  # noqa: PLC0415

    container = PostgresContainer(
        image="pgvector/pgvector:0.8.0-pg16",
        username="postgres",
        password="postgres",
        dbname="resumes",
        driver="psycopg",
    )
    container.start()
    try:
        url = container.get_connection_url()
        if not url.startswith("postgresql+psycopg://"):
            url = url.replace("postgresql://", "postgresql+psycopg://", 1)
        yield url
    finally:
        container.stop()


@pytest.fixture
def configured_env(postgres_container: str, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Set env vars + rebuild engine + ensure pgvector extension is loaded."""
    from sqlalchemy import create_engine, text  # noqa: PLC0415

    jobs_path = ROOT.parent.parent / "data" / "jobs.json"

    monkeypatch.setenv("DATABASE_URL", postgres_container)
    monkeypatch.setenv("JOBS_JSON_PATH", str(jobs_path))
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("APPROACH_OVERRIDE", "")

    eng = create_engine(postgres_container, future=True)
    with eng.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    eng.dispose()

    from src.config import reset_settings_for_test  # noqa: PLC0415
    from src.db.engine import reset_engine_for_test  # noqa: PLC0415
    from src.llm.client import reset_llm_client_for_test  # noqa: PLC0415

    reset_settings_for_test()
    reset_engine_for_test()
    reset_llm_client_for_test()

    yield

    reset_settings_for_test()
    reset_engine_for_test()
    reset_llm_client_for_test()


@pytest.fixture
def app_client(configured_env: None) -> Iterator:
    """Yield a TestClient against the FastAPI app with lifespan run."""
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from src.db.engine import get_engine  # noqa: PLC0415
    from src.db.models import Base  # noqa: PLC0415
    from src.main import app  # noqa: PLC0415

    eng = get_engine()
    Base.metadata.drop_all(bind=eng)

    with TestClient(app) as client:
        yield client
