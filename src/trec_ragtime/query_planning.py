"""Deterministic and precomputed query plans for RAGTIME report requests."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

from ._text_boundaries import continues_after_period, has_paragraph_break
from .fusion import RetrievalQuery
from .topics import TopicRequest


DEFAULT_MULTILINGUAL_PIPELINE = "ragtime2%100"
_FACET_BOUNDARY = re.compile(
    r"(?P<dash>\s+[–—]\s+)|(?<=[.!?;])(?P<space>\s+)"
)


class QueryPlanError(ValueError):
    """Raised when a query plan is missing or malformed."""


class QueryPlanner(Protocol):
    """Structural interface for automatic or precomputed query planners."""

    def plan(self, topic: TopicRequest) -> tuple[RetrievalQuery, ...]: ...


def _topic_query_text(topic: TopicRequest) -> str:
    return "\n".join(
        value.strip()
        for value in (topic.title, topic.background, topic.problem_statement)
        if value.strip()
    )


def _facet_parts(text: str) -> tuple[str, ...]:
    """Split prose, repairing a mid-sentence acronym split when unambiguous."""

    raw_parts: list[tuple[str, bool]] = []
    start = 0
    hard_boundary_before = False
    for match in _FACET_BOUNDARY.finditer(text):
        raw_parts.append((text[start : match.start()], hard_boundary_before))
        is_dash = match.group("dash") is not None
        preceding = "" if is_dash else text[match.start() - 1 : match.start()]
        hard_boundary_before = (
            is_dash
            or preceding != "."
            or has_paragraph_break(match.group(0))
        )
        start = match.end()
    raw_parts.append((text[start:], hard_boundary_before))

    parts: list[str] = []
    for raw_part, hard_boundary in raw_parts:
        part = raw_part.strip()
        if not part:
            continue
        previous = parts[-1] if parts else ""
        if (
            parts
            and not hard_boundary
            and continues_after_period(previous, part)
        ):
            parts[-1] = f"{parts[-1]} {part}"
        else:
            parts.append(part)
    return tuple(parts)


@dataclass(frozen=True, slots=True)
class CompositeQueryPlanner:
    """Produce the submission baseline's single full-request query."""

    pipeline: str = DEFAULT_MULTILINGUAL_PIPELINE

    def __post_init__(self) -> None:
        if not isinstance(self.pipeline, str) or not self.pipeline.strip():
            raise ValueError("pipeline must be a non-empty string")

    def plan(self, topic: TopicRequest) -> tuple[RetrievalQuery, ...]:
        return (
            RetrievalQuery(
                query_id=f"{topic.topic_id}:full",
                text=_topic_query_text(topic),
                pipeline=self.pipeline,
                language="eng",
            ),
        )


@dataclass(frozen=True, slots=True)
class FacetQueryPlanner:
    """Split the problem statement into deterministic retrieval facets.

    This is intentionally a dependency-free control condition. A model-driven
    decomposition can be supplied through :class:`StaticQueryPlanner` without
    changing retrieval, fusion, or submission code.
    """

    pipeline: str = DEFAULT_MULTILINGUAL_PIPELINE
    max_facets: int = 8
    min_facet_characters: int = 24
    include_full_query: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.pipeline, str) or not self.pipeline.strip():
            raise ValueError("pipeline must be a non-empty string")
        if (
            isinstance(self.max_facets, bool)
            or not isinstance(self.max_facets, int)
            or self.max_facets < 1
        ):
            raise ValueError("max_facets must be a positive integer")
        if (
            isinstance(self.min_facet_characters, bool)
            or not isinstance(self.min_facet_characters, int)
            or self.min_facet_characters < 1
        ):
            raise ValueError("min_facet_characters must be a positive integer")
        if not isinstance(self.include_full_query, bool):
            raise ValueError("include_full_query must be a boolean")

    def plan(self, topic: TopicRequest) -> tuple[RetrievalQuery, ...]:
        queries: list[RetrievalQuery] = []
        if self.include_full_query:
            queries.extend(CompositeQueryPlanner(self.pipeline).plan(topic))

        seen: set[str] = set()
        raw_facets = _facet_parts(topic.problem_statement.strip())
        facet_number = 0
        for raw_facet in raw_facets:
            facet = " ".join(raw_facet.split()).strip(" ,;:-")
            canonical = facet.casefold()
            if (
                len(facet) < self.min_facet_characters
                or canonical in seen
                or canonical == topic.title.strip().casefold()
            ):
                continue
            seen.add(canonical)
            facet_number += 1
            query_text = f"{topic.title.strip()}. {facet}"
            queries.append(
                RetrievalQuery(
                    query_id=f"{topic.topic_id}:facet:{facet_number}",
                    text=query_text,
                    pipeline=self.pipeline,
                    language="eng",
                )
            )
            if len(queries) >= self.max_facets + int(self.include_full_query):
                break

        if not queries:
            return (
                RetrievalQuery(
                    query_id=f"{topic.topic_id}:facet:1",
                    text=f"{topic.title.strip()}. {topic.problem_statement.strip()}",
                    pipeline=self.pipeline,
                    language="eng",
                ),
            )
        return tuple(queries)


@dataclass(frozen=True, slots=True)
class StaticQueryPlanner:
    """Return externally prepared multilingual query variants by topic ID."""

    plans: Mapping[str, Sequence[RetrievalQuery]]

    def __post_init__(self) -> None:
        if not isinstance(self.plans, Mapping) or not self.plans:
            raise QueryPlanError("plans must be a non-empty mapping")
        normalized: dict[str, tuple[RetrievalQuery, ...]] = {}
        for topic_id, queries in self.plans.items():
            if not isinstance(topic_id, str) or not topic_id.strip():
                raise QueryPlanError("query-plan topic IDs must be non-empty strings")
            if isinstance(queries, (str, bytes)):
                raise QueryPlanError(f"query plan for topic {topic_id!r} is invalid")
            values = tuple(queries)
            if not values or any(
                not isinstance(query, RetrievalQuery) for query in values
            ):
                raise QueryPlanError(
                    f"query plan for topic {topic_id!r} must contain "
                    "RetrievalQuery values"
                )
            query_ids = [query.query_id for query in values]
            if len(query_ids) != len(set(query_ids)):
                raise QueryPlanError(
                    f"query plan for topic {topic_id!r} contains duplicate query IDs"
                )
            normalized_topic_id = topic_id.strip()
            if normalized_topic_id in normalized:
                raise QueryPlanError(
                    f"query plans contain duplicate topic {normalized_topic_id!r}"
                )
            normalized[normalized_topic_id] = values
        object.__setattr__(self, "plans", MappingProxyType(normalized))

    def plan(self, topic: TopicRequest) -> tuple[RetrievalQuery, ...]:
        try:
            return tuple(self.plans[topic.topic_id])
        except KeyError:
            raise QueryPlanError(
                f"no precomputed query plan for topic {topic.topic_id!r}"
            ) from None


def _required_string(payload: Mapping[str, Any], key: str, *, line: int) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise QueryPlanError(f"query-plan line {line} has invalid {key!r}")
    return value.strip()


def load_query_plans(path: str | Path) -> StaticQueryPlanner:
    """Load one precomputed query-plan object per JSONL line.

    Each object has ``topic_id`` (or ``request_id``) and a non-empty ``queries``
    array. Query entries require ``query_id`` and ``text``; ``pipeline``,
    ``language``, and ``weight`` are optional.
    """

    source = Path(path)
    plans: dict[str, tuple[RetrievalQuery, ...]] = {}
    try:
        handle = source.open("r", encoding="utf-8-sig")
    except OSError as exc:
        raise QueryPlanError(f"unable to open query-plan file: {source}") from exc

    with handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise QueryPlanError(
                    f"query-plan line {line_number} is not valid JSON"
                ) from exc
            if not isinstance(payload, dict):
                raise QueryPlanError(
                    f"query-plan line {line_number} must be a JSON object"
                )

            topic_id = payload.get("topic_id")
            request_id = payload.get("request_id")
            if topic_id is None:
                topic_id = request_id
            elif request_id is not None and request_id != topic_id:
                raise QueryPlanError(
                    f"query-plan line {line_number} has conflicting topic IDs"
                )
            if not isinstance(topic_id, str) or not topic_id.strip():
                raise QueryPlanError(
                    f"query-plan line {line_number} has no valid topic ID"
                )
            topic_id = topic_id.strip()
            if topic_id in plans:
                raise QueryPlanError(
                    f"query-plan line {line_number} repeats topic {topic_id!r}"
                )

            raw_queries = payload.get("queries")
            if not isinstance(raw_queries, list) or not raw_queries:
                raise QueryPlanError(
                    f"query-plan line {line_number} must contain a non-empty "
                    "queries array"
                )
            queries: list[RetrievalQuery] = []
            for index, raw_query in enumerate(raw_queries, start=1):
                if not isinstance(raw_query, dict):
                    raise QueryPlanError(
                        f"query-plan line {line_number} query {index} must be an object"
                    )
                query_id = _required_string(raw_query, "query_id", line=line_number)
                text = _required_string(raw_query, "text", line=line_number)
                pipeline = raw_query.get("pipeline", DEFAULT_MULTILINGUAL_PIPELINE)
                language = raw_query.get("language")
                weight = raw_query.get("weight", 1.0)
                try:
                    query = RetrievalQuery(
                        query_id=query_id,
                        text=text,
                        pipeline=pipeline,
                        language=language,
                        weight=weight,
                    )
                except (TypeError, ValueError) as exc:
                    raise QueryPlanError(
                        f"query-plan line {line_number} query {index} is invalid: {exc}"
                    ) from exc
                queries.append(query)
            plans[topic_id] = tuple(queries)

    if not plans:
        raise QueryPlanError("query-plan file contains no plans")
    return StaticQueryPlanner(plans)


__all__ = [
    "CompositeQueryPlanner",
    "DEFAULT_MULTILINGUAL_PIPELINE",
    "FacetQueryPlanner",
    "QueryPlanError",
    "QueryPlanner",
    "StaticQueryPlanner",
    "load_query_plans",
]
