"""Topic parsing and deterministic Control-A retrieval helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_TOPIC_FIELDS = frozenset(
    {"topic_id", "collection_id", "problem_statement", "background", "limit"}
)


@dataclass(frozen=True, slots=True)
class Topic:
    """One validated RAGTIME topic record."""

    topic_id: str
    collection_id: str
    problem_statement: str
    background: str
    limit: int


def _non_blank_string(value: Any, *, field: str, line_number: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"line {line_number}: {field} must be a non-blank string")
    return value.strip()


def load_topics(path: Path) -> tuple[Topic, ...]:
    """Load and strictly validate ordered topics from a JSONL file."""

    topic_path = Path(path)
    topics: list[Topic] = []
    topic_ids: set[str] = set()
    try:
        handle = topic_path.open("r", encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"unable to read topics file {topic_path}: {exc}") from None

    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"line {line_number}: blank line is not allowed")
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"line {line_number}: invalid JSON ({exc.msg})"
                ) from None
            if not isinstance(payload, dict):
                raise ValueError(f"line {line_number}: topic must be a JSON object")

            fields = set(payload)
            missing = sorted(_TOPIC_FIELDS - fields)
            unexpected = sorted(fields - _TOPIC_FIELDS)
            if missing:
                raise ValueError(
                    f"line {line_number}: missing field {missing[0]!r}"
                )
            if unexpected:
                raise ValueError(
                    f"line {line_number}: unexpected field {unexpected[0]!r}"
                )

            topic_id = _non_blank_string(
                payload["topic_id"], field="topic_id", line_number=line_number
            )
            if topic_id in topic_ids:
                raise ValueError(f"line {line_number}: duplicate topic_id {topic_id!r}")
            topic_ids.add(topic_id)

            collection_id = _non_blank_string(
                payload["collection_id"],
                field="collection_id",
                line_number=line_number,
            )
            problem_statement = _non_blank_string(
                payload["problem_statement"],
                field="problem_statement",
                line_number=line_number,
            )
            background = _non_blank_string(
                payload["background"], field="background", line_number=line_number
            )
            limit = payload["limit"]
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
                raise ValueError(
                    f"line {line_number}: limit must be a positive integer"
                )

            topics.append(
                Topic(
                    topic_id=topic_id,
                    collection_id=collection_id,
                    problem_statement=problem_statement,
                    background=background,
                    limit=limit,
                )
            )

    if not topics:
        raise ValueError(f"topics file is empty: {topic_path}")
    return tuple(topics)


def build_topic_query(topic: Topic) -> str:
    """Build the stable Control-A query from the topic's two text fields."""

    if not isinstance(topic, Topic):
        raise TypeError("topic must be a Topic")
    return f"Problem: {topic.problem_statement}\n\nBackground: {topic.background}"


def pipeline_for_top_k(top_k: int) -> str:
    """Translate the public candidate depth to the organizer pipeline syntax."""

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    return f"ragtime2%{top_k}"


def query_hash(query: str) -> str:
    """Return the SHA-256 digest of the exact UTF-8 query bytes."""

    if not isinstance(query, str) or not query:
        raise ValueError("query must be a non-empty string")
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def _write_atomic(path: Path, content: str) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".control-a-",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def write_control_a_artifact(
    run_dir: Path,
    *,
    manifest: Mapping[str, object],
    records: Iterable[Mapping[str, object]],
) -> None:
    """Atomically write one immutable Control-A run directory."""

    destination = Path(run_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.mkdir(mode=0o700)
    except FileExistsError:
        raise FileExistsError(f"Control-A run already exists: {destination}") from None

    try:
        candidate_lines = "".join(
            json.dumps(
                dict(record),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
            for record in records
        )
        manifest_text = (
            json.dumps(
                dict(manifest),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        _write_atomic(destination / "candidates.jsonl", candidate_lines)
        _write_atomic(destination / "manifest.json", manifest_text)
    except Exception:
        for child in destination.iterdir():
            child.unlink()
        destination.rmdir()
        raise


__all__ = [
    "Topic",
    "build_topic_query",
    "load_topics",
    "pipeline_for_top_k",
    "query_hash",
    "write_control_a_artifact",
]
