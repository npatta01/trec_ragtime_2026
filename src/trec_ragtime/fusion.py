"""Deterministic weighted reciprocal-rank fusion for retrieval results.

RRF intentionally uses only rank positions.  The raw scores emitted by separate
pipelines are retained as provenance, but are never assumed to be calibrated or
comparable across queries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .models import PipelineResult, SearchHit


DEFAULT_RRF_K = 60.0
MAX_FUSED_DOCUMENTS = 1_000


def _require_non_empty_string(value: object, *, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_trimmed_string(value: object, *, field_name: str) -> None:
    _require_non_empty_string(value, field_name=field_name)
    assert isinstance(value, str)
    if value != value.strip():
        raise ValueError(f"{field_name} must not have surrounding whitespace")


def _finite_number(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number")
    try:
        numeric = float(value)
    except OverflowError as error:
        raise ValueError(f"{field_name} must be a finite number") from error
    if not math.isfinite(numeric):
        raise ValueError(f"{field_name} must be a finite number")
    return numeric


def _positive_finite_number(value: object, *, field_name: str) -> float:
    numeric = _finite_number(value, field_name=field_name)
    if numeric <= 0:
        raise ValueError(f"{field_name} must be greater than zero")
    return numeric


def _validated_max_docs(max_docs: object) -> int | None:
    if max_docs is None:
        return None
    if isinstance(max_docs, bool) or not isinstance(max_docs, int):
        raise ValueError("max_docs must be an integer from 1 through 1000")
    if not 1 <= max_docs <= MAX_FUSED_DOCUMENTS:
        raise ValueError("max_docs must be an integer from 1 through 1000")
    return max_docs


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    """One independently executable query in a multi-query retrieval plan."""

    query_id: str
    text: str
    pipeline: str = "ragtime2"
    weight: float = 1.0
    language: str | None = None

    def __post_init__(self) -> None:
        _require_trimmed_string(self.query_id, field_name="query_id")
        _require_non_empty_string(self.text, field_name="text")
        _require_trimmed_string(self.pipeline, field_name="pipeline")
        if self.language is not None:
            _require_trimmed_string(self.language, field_name="language")
        object.__setattr__(
            self,
            "weight",
            _positive_finite_number(self.weight, field_name="weight"),
        )


@dataclass(frozen=True, slots=True)
class QueryRanking:
    """A planned query paired with its exact participant-API result."""

    query: RetrievalQuery
    result: PipelineResult

    def __post_init__(self) -> None:
        if not isinstance(self.query, RetrievalQuery):
            raise ValueError("query must be a RetrievalQuery")
        if not isinstance(self.result, PipelineResult):
            raise ValueError("result must be a PipelineResult")
        if self.result.query != self.query.text:
            raise ValueError("result query does not match RetrievalQuery.text")
        if self.result.pipeline != self.query.pipeline:
            raise ValueError("result pipeline does not match RetrievalQuery.pipeline")


@dataclass(frozen=True, slots=True)
class RankProvenance:
    """One query's rank-only contribution to a fused document score."""

    query_id: str
    query_text: str
    language: str | None
    pipeline: str
    source_rank: int
    source_score: float
    query_weight: float
    contribution: float

    def __post_init__(self) -> None:
        _require_non_empty_string(self.query_id, field_name="query_id")
        _require_non_empty_string(self.query_text, field_name="query_text")
        _require_non_empty_string(self.pipeline, field_name="pipeline")
        if self.language is not None:
            _require_non_empty_string(self.language, field_name="language")
        if isinstance(self.source_rank, bool) or not isinstance(self.source_rank, int):
            raise ValueError("source_rank must be a positive integer")
        if self.source_rank <= 0:
            raise ValueError("source_rank must be a positive integer")
        object.__setattr__(
            self,
            "source_score",
            _finite_number(self.source_score, field_name="source_score"),
        )
        object.__setattr__(
            self,
            "query_weight",
            _positive_finite_number(self.query_weight, field_name="query_weight"),
        )
        object.__setattr__(
            self,
            "contribution",
            _positive_finite_number(self.contribution, field_name="contribution"),
        )


@dataclass(frozen=True, slots=True)
class FusedHit:
    """One document in a fused ranking, with all contributing source ranks."""

    rank: int
    document_id: str
    score: float
    provenance: tuple[RankProvenance, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.rank, bool)
            or not isinstance(self.rank, int)
            or self.rank <= 0
        ):
            raise ValueError("rank must be a positive integer")
        _require_non_empty_string(self.document_id, field_name="document_id")
        object.__setattr__(
            self,
            "score",
            _positive_finite_number(self.score, field_name="score"),
        )
        try:
            provenance = tuple(self.provenance)
        except TypeError as error:
            raise ValueError(
                "provenance must be an iterable of RankProvenance"
            ) from error
        if not provenance:
            raise ValueError("provenance must not be empty")
        if any(not isinstance(item, RankProvenance) for item in provenance):
            raise ValueError("provenance must contain only RankProvenance values")
        query_ids = tuple(item.query_id for item in provenance)
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("provenance query IDs must be unique per document")
        object.__setattr__(self, "provenance", provenance)


@dataclass(frozen=True, slots=True)
class FusedRanking:
    """A deterministic weighted-RRF result and the queries that produced it."""

    queries: tuple[RetrievalQuery, ...]
    hits: tuple[FusedHit, ...]
    k: float
    max_docs: int | None = None

    def __post_init__(self) -> None:
        try:
            queries = tuple(self.queries)
        except TypeError as error:
            raise ValueError("queries must be an iterable of RetrievalQuery") from error
        if not queries:
            raise ValueError("queries must not be empty")
        if any(not isinstance(query, RetrievalQuery) for query in queries):
            raise ValueError("queries must contain only RetrievalQuery values")
        query_ids = tuple(query.query_id for query in queries)
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("query IDs must be unique")

        try:
            hits = tuple(self.hits)
        except TypeError as error:
            raise ValueError("hits must be an iterable of FusedHit") from error
        if any(not isinstance(hit, FusedHit) for hit in hits):
            raise ValueError("hits must contain only FusedHit values")
        if tuple(hit.rank for hit in hits) != tuple(range(1, len(hits) + 1)):
            raise ValueError("fused hit ranks must be consecutive starting at one")
        document_ids = tuple(hit.document_id for hit in hits)
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("fused document IDs must be unique")
        expected_order = tuple(
            sorted(hits, key=lambda hit: (-hit.score, hit.document_id))
        )
        if hits != expected_order:
            raise ValueError("fused hits must be sorted by score then document ID")

        query_positions = {query_id: index for index, query_id in enumerate(query_ids)}
        for hit in hits:
            if any(item.query_id not in query_positions for item in hit.provenance):
                raise ValueError("provenance refers to an unknown query ID")
            provenance_order = tuple(
                (query_positions[item.query_id], item.source_rank)
                for item in hit.provenance
            )
            if provenance_order != tuple(sorted(provenance_order)):
                raise ValueError(
                    "provenance must follow query input order and source rank"
                )

        validated_max_docs = _validated_max_docs(self.max_docs)
        if validated_max_docs is not None and len(hits) > validated_max_docs:
            raise ValueError("hits exceed max_docs")
        object.__setattr__(self, "queries", queries)
        object.__setattr__(self, "hits", hits)
        object.__setattr__(
            self,
            "k",
            _positive_finite_number(self.k, field_name="k"),
        )
        object.__setattr__(self, "max_docs", validated_max_docs)

    @property
    def query_ids(self) -> tuple[str, ...]:
        """Return query identifiers in the fusion input order."""

        return tuple(query.query_id for query in self.queries)


def _deduplicated_source_hits(hits: tuple[SearchHit, ...]) -> tuple[SearchHit, ...]:
    """Return each document's best source rank without consulting raw scores."""

    validated: list[tuple[int, SearchHit]] = []
    for position, hit in enumerate(hits):
        if not isinstance(hit, SearchHit):
            raise ValueError("PipelineResult.hits must contain only SearchHit values")
        if isinstance(hit.rank, bool) or not isinstance(hit.rank, int) or hit.rank <= 0:
            raise ValueError("source hit rank must be a positive integer")
        _require_non_empty_string(hit.document_id, field_name="source document_id")
        _finite_number(hit.score, field_name="source score")
        validated.append((position, hit))

    # The source rank is authoritative.  Original position resolves malformed
    # equal-rank duplicates deterministically without using uncalibrated scores.
    validated.sort(key=lambda item: (item[1].rank, item[0]))
    seen: set[str] = set()
    unique: list[SearchHit] = []
    for _, hit in validated:
        if hit.document_id in seen:
            continue
        seen.add(hit.document_id)
        unique.append(hit)
    return tuple(unique)


def weighted_reciprocal_rank_fusion(
    rankings: Iterable[QueryRanking],
    *,
    k: float = DEFAULT_RRF_K,
    max_docs: int | None = None,
) -> FusedRanking:
    """Fuse query rankings using ``sum(weight / (k + source_rank))``.

    Source scores are retained only in :class:`RankProvenance`; they never
    affect the fused score or ordering because separate retrieval runs are not
    assumed to share a calibrated score scale.
    """

    validated_k = _positive_finite_number(k, field_name="k")
    validated_max_docs = _validated_max_docs(max_docs)
    try:
        ranking_items = tuple(rankings)
    except TypeError as error:
        raise ValueError("rankings must be an iterable of QueryRanking") from error
    if not ranking_items:
        raise ValueError("rankings must not be empty")
    if any(not isinstance(ranking, QueryRanking) for ranking in ranking_items):
        raise ValueError("rankings must contain only QueryRanking values")

    queries = tuple(ranking.query for ranking in ranking_items)
    query_ids = tuple(query.query_id for query in queries)
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("query IDs must be unique")

    contributions: dict[str, list[float]] = {}
    provenance_by_document: dict[str, list[RankProvenance]] = {}
    for ranking in ranking_items:
        query = ranking.query
        for hit in _deduplicated_source_hits(ranking.result.hits):
            contribution = query.weight / (validated_k + hit.rank)
            contributions.setdefault(hit.document_id, []).append(contribution)
            provenance_by_document.setdefault(hit.document_id, []).append(
                RankProvenance(
                    query_id=query.query_id,
                    query_text=query.text,
                    language=query.language,
                    pipeline=query.pipeline,
                    source_rank=hit.rank,
                    source_score=hit.score,
                    query_weight=query.weight,
                    contribution=contribution,
                )
            )

    scored_documents: list[tuple[str, float]] = []
    for document_id, document_contributions in contributions.items():
        try:
            score = math.fsum(document_contributions)
        except OverflowError as error:
            raise ValueError("fused score must be finite") from error
        scored_documents.append(
            (
                document_id,
                _positive_finite_number(score, field_name="fused score"),
            )
        )
    scored_documents.sort(key=lambda item: (-item[1], item[0]))
    if validated_max_docs is not None:
        scored_documents = scored_documents[:validated_max_docs]

    fused_hits = tuple(
        FusedHit(
            rank=rank,
            document_id=document_id,
            score=score,
            provenance=tuple(provenance_by_document[document_id]),
        )
        for rank, (document_id, score) in enumerate(scored_documents, start=1)
    )
    return FusedRanking(
        queries=queries,
        hits=fused_hits,
        k=validated_k,
        max_docs=validated_max_docs,
    )


def fuse_rankings(
    rankings: Iterable[QueryRanking],
    *,
    k: float = DEFAULT_RRF_K,
    max_docs: int | None = None,
) -> FusedRanking:
    """Concise alias for :func:`weighted_reciprocal_rank_fusion`."""

    return weighted_reciprocal_rank_fusion(rankings, k=k, max_docs=max_docs)


__all__ = [
    "DEFAULT_RRF_K",
    "MAX_FUSED_DOCUMENTS",
    "FusedHit",
    "FusedRanking",
    "QueryRanking",
    "RankProvenance",
    "RetrievalQuery",
    "fuse_rankings",
    "weighted_reciprocal_rank_fusion",
]
