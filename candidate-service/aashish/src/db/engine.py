"""SQLAlchemy engine + session factory.

We use a synchronous engine throughout. FastAPI is async but our DB calls are
short-lived and pgvector cosine queries return in well under 30ms; running
SQLAlchemy in sync mode keeps the codebase smaller and avoids the async-greenlet
machinery for what is really a CPU-bound + LLM-IO-bound workload.
"""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.config import get_settings


class Base(DeclarativeBase):
    pass


_engine: Engine | None = None
_SessionFactory: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_engine(
            settings.database_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=5,
            future=True,
        )
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)
    return _SessionFactory


def SessionLocal() -> Session:  # noqa: N802 - factory style
    return get_session_factory()()


def reset_engine_for_test() -> None:
    """Test helper: drop cached engine/session factory so a fresh DATABASE_URL is picked up."""
    global _engine, _SessionFactory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionFactory = None
