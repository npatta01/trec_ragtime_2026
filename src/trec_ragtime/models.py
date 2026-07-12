"""Typed response models for retrieval and document hydration."""

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


@dataclass(frozen=True, slots=True)
class HuggingFaceDocument:
    """One original-language RAGTIME2 row loaded from Hugging Face."""

    dataset: str
    config: str
    split: str
    document_id: str
    text: str
    url: str | None
    date: str | None
    local_cache_hit: bool
    cache_stored_at: float | None

    @property
    def language(self) -> str:
        """Return the corpus language encoded by the source split."""

        return self.split


@dataclass(frozen=True, slots=True)
class HydrationFailure:
    """A sanitized Hugging Face split lookup failure."""

    split: str
    message: str


@dataclass(frozen=True, slots=True)
class HydrationResult:
    """Documents found by a bounded batch hydration request."""

    documents: tuple[HuggingFaceDocument, ...]
    unresolved_ids: tuple[str, ...]
    failures: tuple[HydrationFailure, ...]

    def by_id(self) -> dict[str, HuggingFaceDocument]:
        """Index hydrated documents by their opaque corpus ID."""

        return {document.document_id: document for document in self.documents}


@dataclass(frozen=True, slots=True)
class RetrievedDocument:
    """One ranked hit, optionally hydrated with document content."""

    rank: int
    document_id: str
    score: float
    text: str | None
    url: str | None
    date: str | None
    language: str | None
    content_source: str | None
    content_local_cache_hit: bool | None

    @property
    def hydrated(self) -> bool:
        return self.text is not None

    def as_record(self, *, include_text: bool = True) -> dict[str, Any]:
        record: dict[str, Any] = {
            "rank": self.rank,
            "document_id": self.document_id,
            "score": self.score,
            "url": self.url,
            "date": self.date,
            "language": self.language,
            "content_source": self.content_source,
            "content_local_cache_hit": self.content_local_cache_hit,
        }
        if include_text:
            record["text"] = self.text
        return record


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """A ranked search response joined to corpus content where available."""

    search_result: PipelineResult
    documents: tuple[RetrievedDocument, ...]
    hydration_failures: tuple[HydrationFailure, ...] = ()

    @property
    def query(self) -> str:
        return self.search_result.query

    @property
    def pipeline(self) -> str:
        return self.search_result.pipeline

    @property
    def search_local_cache_hit(self) -> bool:
        return self.search_result.local_cache_hit

    @property
    def fully_hydrated(self) -> bool:
        return all(document.hydrated for document in self.documents)

    @property
    def unresolved_ids(self) -> tuple[str, ...]:
        return tuple(
            document.document_id for document in self.documents if not document.hydrated
        )

    def as_records(self, *, include_text: bool = True) -> list[dict[str, Any]]:
        return [
            document.as_record(include_text=include_text) for document in self.documents
        ]
