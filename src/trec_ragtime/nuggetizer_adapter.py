"""The only boundary to the upstream Nuggetizer package.

Nuggetization is delegated to ``nuggetizer``; this module owns nothing but the
call shape, so a package change cannot leak into the pipeline.

Provider routing needs no wrapper: upstream builds ``OpenAI(api_key=...)`` with
no explicit base URL, so the SDK resolves ``OPENAI_API_KEY`` and, when set,
``OPENAI_BASE_URL``. OpenRouter and a LiteLLM proxy are both reachable that way.

``BoundedHandler`` exists because upstream's ``LLMHandler.run`` retries forever:
it catches every exception, rotates the key, sleeps, and loops with no exit
condition, so one bad key hangs a whole batch instead of failing a topic. A
finite budget turns that into an error, which the artifact contract requires --
a timed-out result must never become a successful cache entry.

Upstream exposes no seed, so determinism comes from the artifact cache rather
than from the provider.
"""

from __future__ import annotations

import importlib.metadata
from collections.abc import Mapping, Sequence
from typing import Any

import msgspec
from nuggetizer.core.llm import LLMHandler
from nuggetizer.core.types import Document, Query, Request, ScoredNugget
from nuggetizer.models.nuggetizer import Nuggetizer


PACKAGE_NAME = "nuggetizer"
MAX_COMPLETION_TOKENS = 2048
REQUEST_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_CALLS = 64
_IMPORTANCE_VALUES = frozenset({"vital", "okay"})
_ASSIGNMENT_VALUES = frozenset({"support", "partial_support", "not_support"})


class NuggetizerError(RuntimeError):
    """Raised when the upstream package is unusable or returns invalid output."""


class NuggetizerConfig(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Result-affecting upstream settings, hashed into the artifact fingerprint."""

    model: str
    max_nuggets: int = 20
    creator_window_size: int = 10
    scorer_window_size: int = 20
    max_calls: int = DEFAULT_MAX_CALLS


class BoundedHandler(LLMHandler):
    """An ``LLMHandler`` with a finite call budget shared across one topic."""

    def __init__(self, model: str, *, max_calls: int) -> None:
        super().__init__(model)
        if isinstance(max_calls, bool) or not isinstance(max_calls, int) or max_calls < 1:
            raise ValueError("max_calls must be a positive integer")
        self.remaining = max_calls
        self.error: Exception | None = None

    def run(
        self, messages: list[dict[str, str]], temperature: float = 0
    ) -> tuple[str, int]:
        if self.remaining <= 0:
            # Surface the provider failure that consumed the budget; otherwise a
            # bad key is reported only as an exhausted budget.
            raise NuggetizerError(
                "nuggetizer provider call budget is exhausted"
                + (f" after {type(self.error).__name__}" if self.error else "")
            ) from self.error
        self.remaining -= 1
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            content = completion.choices[0].message.content
        except Exception as error:  # noqa: BLE001 - recorded, then re-raised
            self.error = error
            raise
        if not isinstance(content, str) or not content.strip():
            raise NuggetizerError("nuggetizer provider returned an empty completion")
        # Upstream discards the token count, so it is not worth computing.
        return content, 0


def package_identity() -> Mapping[str, str]:
    """Return the exact installed package revision for the fingerprint."""

    distribution = importlib.metadata.distribution(PACKAGE_NAME)
    identity = {"name": PACKAGE_NAME, "version": distribution.version}
    direct_url = distribution.read_text("direct_url.json")
    if direct_url is not None:
        commit = msgspec.json.decode(direct_url).get("vcs_info", {}).get("commit_id")
        if isinstance(commit, str) and commit:
            identity["commit"] = commit
    return identity


def build(config: NuggetizerConfig) -> Nuggetizer:
    """Return a configured Nuggetizer whose provider calls are bounded."""

    nuggetizer = Nuggetizer(
        model=config.model,
        creator_window_size=config.creator_window_size,
        scorer_window_size=config.scorer_window_size,
        # One assign call per passage, rather than one per window of nuggets.
        assigner_window_size=config.max_nuggets,
        max_nuggets=config.max_nuggets,
        log_level=0,
    )
    handler = BoundedHandler(config.model, max_calls=config.max_calls)
    nuggetizer.creator_llm = handler
    nuggetizer.scorer_llm = handler
    nuggetizer.assigner_llm = handler
    return nuggetizer


class NuggetizerBackend:
    """The nuggetization operations the stage is allowed to use."""

    def __init__(self, config: NuggetizerConfig, *, nuggetizer: Any | None = None) -> None:
        self.config = config
        self._nuggetizer = build(config) if nuggetizer is None else nuggetizer

    def create(
        self, *, topic_id: str, query: str, passages: Sequence[tuple[str, str]]
    ) -> tuple[tuple[str, str], ...]:
        """Return ``(claim, importance)`` for one topic's complete passage set."""

        request = Request(
            query=Query(qid=topic_id, text=query),
            documents=[Document(docid=alias, segment=text) for alias, text in passages],
        )
        return tuple(
            (_text(nugget), _label(nugget.importance, _IMPORTANCE_VALUES, "importance"))
            for nugget in self._nuggetizer.create(request)
        )

    def assign(
        self, *, query: str, passage_text: str, nuggets: Sequence[tuple[str, str]]
    ) -> tuple[str, ...]:
        """Return one support label per nugget for a single passage."""

        scored = [
            ScoredNugget(text=claim, importance=importance)
            for claim, importance in nuggets
        ]
        assigned = self._nuggetizer.assign(query, passage_text, scored)
        if len(assigned) != len(scored):
            raise NuggetizerError("nuggetizer returned an incomplete assignment set")
        for nugget, expected in zip(assigned, scored, strict=True):
            # Linkage is positional, so a silent reorder would mislink evidence.
            if _text(nugget) != expected.text:
                raise NuggetizerError("nuggetizer reordered or rewrote a nugget claim")
        return tuple(
            _label(nugget.assignment, _ASSIGNMENT_VALUES, "assignment")
            for nugget in assigned
        )


def _text(nugget: Any) -> str:
    if not isinstance(nugget.text, str) or not nugget.text.strip():
        raise NuggetizerError("nuggetizer returned an empty claim")
    return nugget.text


def _label(value: Any, allowed: frozenset[str], field: str) -> str:
    # Upstream degrades to a "failed" sentinel instead of raising; that is an
    # error here, never a stored value.
    if value not in allowed:
        raise NuggetizerError(f"nuggetizer returned an unusable {field}")
    return value


__all__ = [
    "DEFAULT_MAX_CALLS",
    "MAX_COMPLETION_TOKENS",
    "BoundedHandler",
    "NuggetizerBackend",
    "NuggetizerConfig",
    "NuggetizerError",
    "build",
    "package_identity",
]
