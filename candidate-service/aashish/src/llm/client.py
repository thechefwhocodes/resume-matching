"""Single OpenAI client wrapper used for embeddings, extraction and rerank.

The client is deliberately minimal: it owns the OpenAI SDK handle, batches
embedding requests, supports structured chat completions, and tracks per-call
token usage + USD cost so callers can surface this in MatchMetadata.

If no API key is configured the client is `is_available == False`. In that mode
`embed()` returns deterministic zero-vectors so ingest still works, and
`chat_structured()` raises -- callers must guard with `is_available`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import Settings, get_settings

if TYPE_CHECKING:  # pragma: no cover
    from openai import OpenAI

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# Per-1K-token USD prices for the models we ship with. Used purely for the
# `metadata.cost_usd` field; not load-bearing. Keep in sync with OpenAI pricing
# (snapshot: late 2025).
_PRICING_PER_1K = {
    "text-embedding-3-small": {"embedding": 0.00002},
    "text-embedding-3-large": {"embedding": 0.00013},
    "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
    "gpt-4o": {"prompt": 0.0025, "completion": 0.01},
}


@dataclass
class CostTracker:
    """Accumulates token + USD usage for a single request."""

    prompt: int = 0
    completion: int = 0
    embedding: int = 0
    cost_usd: float = 0.0
    _entries: list[dict[str, Any]] = field(default_factory=list)

    def add_chat(self, model: str, prompt: int, completion: int) -> None:
        self.prompt += prompt
        self.completion += completion
        prices = _PRICING_PER_1K.get(model, {})
        self.cost_usd += (prompt / 1000.0) * prices.get("prompt", 0.0)
        self.cost_usd += (completion / 1000.0) * prices.get("completion", 0.0)
        self._entries.append(
            {"kind": "chat", "model": model, "prompt": prompt, "completion": completion}
        )

    def add_embedding(self, model: str, tokens: int) -> None:
        self.embedding += tokens
        prices = _PRICING_PER_1K.get(model, {})
        self.cost_usd += (tokens / 1000.0) * prices.get("embedding", 0.0)
        self._entries.append({"kind": "embedding", "model": model, "tokens": tokens})

    def merge(self, other: CostTracker) -> None:
        self.prompt += other.prompt
        self.completion += other.completion
        self.embedding += other.embedding
        self.cost_usd += other.cost_usd
        self._entries.extend(other._entries)


class LLMClient:
    """Wraps the OpenAI SDK with structured output + cost tracking helpers."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._client: OpenAI | None = None
        if self.settings.has_openai_key:
            from openai import OpenAI  # noqa: PLC0415

            self._client = OpenAI(api_key=self.settings.openai_api_key)

    @property
    def is_available(self) -> bool:
        return self._client is not None

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def embed(
        self,
        texts: list[str],
        *,
        tracker: CostTracker | None = None,
    ) -> list[list[float]]:
        """Embed a batch of strings. Returns zero-vectors when no key is set.

        OpenAI's batch embeddings endpoint accepts up to ~2048 items per call;
        we chunk defensively at 256 to stay well under request-size limits.
        """
        dim = self.settings.embedding_dim
        if not self.is_available:
            return [[0.0] * dim for _ in texts]

        out: list[list[float]] = []
        batch_size = 256
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            resp = self._embed_chunk(chunk)
            out.extend([d.embedding for d in resp.data])
            if tracker is not None and resp.usage is not None:
                tracker.add_embedding(self.settings.embed_model, resp.usage.total_tokens)
        return out

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _embed_chunk(self, chunk: list[str]):
        assert self._client is not None
        return self._client.embeddings.create(
            model=self.settings.embed_model,
            input=chunk,
        )

    # ------------------------------------------------------------------
    # Chat (structured outputs)
    # ------------------------------------------------------------------

    def chat_structured(
        self,
        *,
        response_model: type[T],
        system: str,
        user: str,
        model: str | None = None,
        temperature: float = 0.0,
        seed: int | None = 42,
        tracker: CostTracker | None = None,
    ) -> T:
        """Issue a structured chat completion and parse into `response_model`."""
        if not self.is_available:
            raise RuntimeError(
                "LLMClient.chat_structured called without OPENAI_API_KEY. "
                "Caller must guard with `client.is_available`."
            )

        chosen_model = model or self.settings.extract_model
        completion = self._chat_call(
            model=chosen_model,
            system=system,
            user=user,
            response_model=response_model,
            temperature=temperature,
            seed=seed,
        )

        if tracker is not None and completion.usage is not None:
            tracker.add_chat(
                chosen_model,
                prompt=completion.usage.prompt_tokens or 0,
                completion=completion.usage.completion_tokens or 0,
            )

        msg = completion.choices[0].message
        parsed = getattr(msg, "parsed", None)
        if parsed is None:
            raise RuntimeError(
                f"OpenAI structured output returned no parsed content for {response_model.__name__}"
            )
        return parsed  # type: ignore[no-any-return]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _chat_call(
        self,
        *,
        model: str,
        system: str,
        user: str,
        response_model: type[BaseModel],
        temperature: float,
        seed: int | None,
    ):
        assert self._client is not None
        return self._client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format=response_model,
            temperature=temperature,
            seed=seed,
        )


_singleton: LLMClient | None = None


def get_llm_client() -> LLMClient:
    global _singleton
    if _singleton is None:
        _singleton = LLMClient()
    return _singleton


def reset_llm_client_for_test() -> None:
    global _singleton
    _singleton = None
