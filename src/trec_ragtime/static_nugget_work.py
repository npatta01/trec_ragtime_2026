"""Assemble static-plan rerankings for the existing nugget runner."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

import msgspec

from .nugget_cache import NuggetInput
from .nuggets_runner import TopicWork, evidence_from_reranking


@dataclass(frozen=True, slots=True)
class StaticRerankingInput:
    facet_index: int
    query: str
    object_key: str
    request_digest: bytes
    source_fingerprint: str
    result: Any


def build_static_topic_work(
    *,
    topic_id: str,
    narrative: str,
    representation: str,
    rows: Sequence[StaticRerankingInput],
) -> TopicWork:
    """Carry exact per-query reranking lineage into the proven nugget stage."""

    items = tuple(rows)
    if not topic_id or not narrative or not items:
        raise ValueError("topic ID, narrative, and reranking rows are required")
    if representation not in {"canonical", "report_text"}:
        raise ValueError("representation is invalid")
    if len({row.facet_index for row in items}) != len(items):
        raise ValueError("facet indices must be unique")
    for row in items:
        if (
            not isinstance(row.facet_index, int)
            or row.facet_index < 0
            or not row.query
            or not row.object_key
            or not row.source_fingerprint
            or not isinstance(row.request_digest, bytes)
            or len(row.request_digest) != 32
            or getattr(row.result, "query", None) != row.query
            or getattr(row.result, "representation", None) != representation
        ):
            raise ValueError("static reranking input is invalid")
    source_digest = hashlib.sha256(
        msgspec.msgpack.encode(
            (representation, tuple(row.source_fingerprint for row in items))
        )
    ).hexdigest()
    return TopicWork(
        topic_id=topic_id,
        query=narrative,
        representation=representation,
        facets=evidence_from_reranking(
            tuple(row.result for row in items),
            queries=tuple(row.query for row in items),
            facet_indices=tuple(row.facet_index for row in items),
        ),
        inputs=tuple(
            NuggetInput(
                row.facet_index,
                row.query,
                row.object_key,
                row.request_digest,
            )
            for row in items
        ),
        passage_source_fingerprint=f"static-plan-reranking-v1:{source_digest}",
        retrievals=tuple(row.result for row in items),
    )


__all__ = ["StaticRerankingInput", "build_static_topic_work"]
