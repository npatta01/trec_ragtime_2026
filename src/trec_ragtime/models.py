"""Typed response models for the RAGTIME API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One deterministically ranked search result."""

    rank: int
    document_id: str
    score: float

    def as_record(self) -> dict[str, int | str | float]:
        return {"rank": self.rank, "document_id": self.document_id, "score": self.score}


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Validated response from ``POST /pipeline``."""

    query: str
    pipeline: str
    hits: tuple[SearchHit, ...]
    processed: bool | None
    local_cache_hit: bool
    remote_cached: bool | None
    server_timestamp: float | None
    cache_stored_at: float | None
    extra: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def as_records(self) -> list[dict[str, int | str | float]]:
        return [hit.as_record() for hit in self.hits]


@dataclass(frozen=True, slots=True)
class Document:
    """Validated response from ``POST /content``."""

    collection: str
    document_id: str
    text: str
    local_cache_hit: bool
    remote_cached: bool | None
    server_timestamp: float | None
    cache_stored_at: float | None
    extra: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
