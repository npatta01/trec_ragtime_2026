"""Validated topic requests for the TREC RAGTIME 2026 collection."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any


RAGTIME_2026_COLLECTION_ID = "ragtime/2/all"

_KNOWN_FIELDS = frozenset(
    {
        "topic_id",
        "request_id",
        "collection_id",
        "title",
        "problem_statement",
        "background",
        "limit",
    }
)


class TopicFileError(ValueError):
    """A sanitized, line-oriented error in a topic JSONL file."""


@dataclass(frozen=True, slots=True)
class TopicRequest:
    """One validated 2026 report request, canonicalized to ``topic_id``."""

    topic_id: str
    collection_id: str
    title: str
    problem_statement: str
    background: str
    limit: int
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        _validate_nonempty_string("topic_id", self.topic_id)
        if self.topic_id != self.topic_id.strip():
            raise ValueError("topic_id must not have leading or trailing whitespace")
        if self.collection_id != RAGTIME_2026_COLLECTION_ID:
            raise ValueError(
                f"collection_id must be {RAGTIME_2026_COLLECTION_ID!r}"
            )
        _validate_nonempty_string("title", self.title)
        _validate_nonempty_string("problem_statement", self.problem_statement)
        _validate_nonempty_string("background", self.background)
        if isinstance(self.limit, bool) or not isinstance(self.limit, int):
            raise ValueError("limit must be a positive integer")
        if self.limit <= 0:
            raise ValueError("limit must be a positive integer")
        if not isinstance(self.extra, Mapping):
            raise ValueError("extra must be a mapping")
        if any(not isinstance(key, str) for key in self.extra):
            raise ValueError("extra field names must be strings")
        if _KNOWN_FIELDS.intersection(self.extra):
            raise ValueError("extra must not contain a reserved topic field")

        # Copy caller-owned values and expose a stable, read-only top-level mapping.
        frozen_extra = MappingProxyType(
            {
                key: deepcopy(self.extra[key])
                for key in sorted(self.extra)
            }
        )
        object.__setattr__(self, "extra", frozen_extra)

    @classmethod
    def from_mapping(cls, record: Mapping[str, Any]) -> TopicRequest:
        """Validate a JSON-like mapping and canonicalize its topic ID alias."""

        if not isinstance(record, Mapping):
            raise ValueError("topic must be an object")
        if any(not isinstance(key, str) for key in record):
            raise ValueError("topic field names must be strings")

        has_topic_id = "topic_id" in record
        has_request_id = "request_id" in record
        if not has_topic_id and not has_request_id:
            raise ValueError("missing required field topic_id (or request_id)")
        if (
            has_topic_id
            and has_request_id
            and record["topic_id"] != record["request_id"]
        ):
            raise ValueError("topic_id and request_id conflict")
        topic_id = record["topic_id"] if has_topic_id else record["request_id"]

        required = {
            name: _required_field(record, name)
            for name in (
                "collection_id",
                "title",
                "problem_statement",
                "background",
                "limit",
            )
        }
        extra = {
            key: record[key]
            for key in sorted(record)
            if key not in _KNOWN_FIELDS
        }
        return cls(topic_id=topic_id, extra=extra, **required)

    def as_record(self) -> dict[str, Any]:
        """Return a deterministic JSON-serializable record using ``topic_id``."""

        record: dict[str, Any] = {
            "topic_id": self.topic_id,
            "collection_id": self.collection_id,
            "title": self.title,
            "problem_statement": self.problem_statement,
            "background": self.background,
            "limit": self.limit,
        }
        record.update(
            (key, deepcopy(self.extra[key]))
            for key in sorted(self.extra)
        )
        return record


def load_topic_requests(path: str | Path) -> tuple[TopicRequest, ...]:
    """Load a topic JSONL file in source order with sanitized line errors."""

    topics: list[TopicRequest] = []
    first_line_by_id: dict[str, int] = {}
    topic_path = Path(path)

    try:
        handle = topic_path.open("r", encoding="utf-8-sig")
    except OSError as error:
        raise TopicFileError(f"unable to open topic file: {topic_path}") from error

    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line, parse_constant=_reject_json_constant)
            except (json.JSONDecodeError, ValueError):
                raise TopicFileError(
                    f"line {line_number}: invalid JSON"
                ) from None
            if not isinstance(record, dict):
                raise TopicFileError(
                    f"line {line_number}: topic must be a JSON object"
                )
            try:
                topic = TopicRequest.from_mapping(record)
            except (TypeError, ValueError) as exc:
                raise TopicFileError(
                    f"line {line_number}: invalid topic: {exc}"
                ) from None

            first_line = first_line_by_id.get(topic.topic_id)
            if first_line is not None:
                raise TopicFileError(
                    f"line {line_number}: duplicate topic_id "
                    f"(first defined on line {first_line})"
                )
            first_line_by_id[topic.topic_id] = line_number
            topics.append(topic)

    if not topics:
        raise TopicFileError("topic file is empty")
    return tuple(topics)


# A concise compatibility spelling for notebook callers.
load_topics = load_topic_requests


def _required_field(record: Mapping[str, Any], name: str) -> Any:
    try:
        return record[name]
    except KeyError:
        raise ValueError(f"missing required field {name}") from None


def _validate_nonempty_string(name: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _reject_json_constant(_value: str) -> Any:
    raise ValueError("non-finite JSON number")


__all__ = [
    "RAGTIME_2026_COLLECTION_ID",
    "TopicFileError",
    "TopicRequest",
    "load_topic_requests",
    "load_topics",
]
