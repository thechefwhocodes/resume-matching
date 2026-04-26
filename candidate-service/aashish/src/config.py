"""Centralised configuration loaded from environment / .env via pydantic-settings."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ApproachName = Literal["bm25", "bm25+rerank", "embed", "embed+rerank"]


def _default_jobs_path() -> str:
    here = Path(__file__).resolve().parent.parent
    return str((here / ".." / ".." / "data" / "jobs.json").resolve())


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", "../../.env"],
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        env_ignore_empty=True,
    )

    database_url: str = Field(
        default="postgresql+psycopg://postgres:postgres@localhost:5432/resumes",
        description="SQLAlchemy URL for Postgres",
    )

    openai_api_key: str | None = Field(default=None)
    embed_model: str = Field(default="text-embedding-3-small")
    extract_model: str = Field(default="gpt-4o-mini")
    rerank_model: str = Field(default="gpt-4o-mini")

    approach_override: ApproachName | None = Field(default=None)

    jobs_json_path: str = Field(default_factory=_default_jobs_path)

    retrieval_top_k: int = Field(default=30, ge=5, le=100)
    result_top_k: int = Field(default=10, ge=1, le=50)
    rerank_min_score: int = Field(default=50, ge=0, le=100)

    embedding_dim: int = Field(default=1536)

    @property
    def has_openai_key(self) -> bool:
        return bool(self.openai_api_key and self.openai_api_key.strip())

    def default_approach(self) -> ApproachName:
        if self.approach_override:
            return self.approach_override
        return "embed+rerank" if self.has_openai_key else "bm25"


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings_for_test() -> None:
    """Test helper: forces settings to be reloaded on next get_settings() call."""
    global _settings
    _settings = None
