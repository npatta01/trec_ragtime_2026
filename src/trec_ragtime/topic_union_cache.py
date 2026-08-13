"""Typed durable cache for topic-level maximum-need scoring results."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any

from .topic_union_scoring import (
    InformationNeed,
    RetrievalSource,
    ScoredTopicDocument,
    TopicScoreResult,
)


OBJECT_ROOT = "topic-scores/v1"
SCHEMA_VERSION = 1
_HEX64 = re.compile(r"[0-9a-f]{64}")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-blank string")
    return value


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class TopicScoreFingerprint:
    scorer_fingerprint: str
    chunker_fingerprint: str
    representation: str
    scoring_policy: str = "max-need-passage-v1"
    tie_policy: str = "document-id-tie-v1"
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("scorer_fingerprint", "chunker_fingerprint"):
            _text(getattr(self, name), name)
        if self.representation not in {"canonical", "report_text"}:
            raise ValueError("invalid representation")
        if self.scoring_policy != "max-need-passage-v1":
            raise ValueError("unsupported scoring policy")
        if self.tie_policy != "document-id-tie-v1":
            raise ValueError("unsupported tie policy")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported topic-score schema")

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            _canonical(
                {
                    "schema_version": self.schema_version,
                    "scorer_fingerprint": self.scorer_fingerprint,
                    "chunker_fingerprint": self.chunker_fingerprint,
                    "representation": self.representation,
                    "scoring_policy": self.scoring_policy,
                    "tie_policy": self.tie_policy,
                }
            )
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class TopicScoreRequest:
    topic_id: str
    plan_sha256: str
    needs: tuple[InformationNeed, ...]
    input_object_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.topic_id, "topic_id")
        if not isinstance(self.plan_sha256, str) or not _HEX64.fullmatch(self.plan_sha256):
            raise ValueError("plan_sha256 must be a lowercase SHA-256")
        if not self.needs or len({row.need_id for row in self.needs}) != len(self.needs):
            raise ValueError("needs must be nonempty with unique IDs")
        if (
            not self.input_object_keys
            or tuple(sorted(set(self.input_object_keys))) != self.input_object_keys
        ):
            raise ValueError("input_object_keys must be sorted and unique")
        for key in self.input_object_keys:
            _text(key, "input object key")

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            _canonical(
                {
                    "topic_id": self.topic_id,
                    "plan_sha256": self.plan_sha256,
                    "needs": [
                        {"need_id": row.need_id, "text": row.text} for row in self.needs
                    ],
                    "input_object_keys": list(self.input_object_keys),
                }
            )
        ).hexdigest()


def _result_record(result: TopicScoreResult) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "topic_id": result.topic_id,
        "representation": result.representation,
        "needs": [
            {"need_id": row.need_id, "text": row.text} for row in result.needs
        ],
        "input_object_keys": list(result.input_object_keys),
        "unique_document_count": result.unique_document_count,
        "passage_count": result.passage_count,
        "scorer_fingerprint": result.scorer_fingerprint,
        "chunker_fingerprint": result.chunker_fingerprint,
        "documents": [
            {
                "document_id": row.document_id,
                "score": row.score,
                "winning_need_id": row.winning_need_id,
                "passage_start": row.passage_start,
                "passage_end": row.passage_end,
                "passage_content_hash": row.passage_content_hash.hex(),
                "sources": [
                    {
                        "task_id": source.task_id,
                        "query": source.query,
                        "object_key": source.object_key,
                        "original_rank": source.original_rank,
                        "original_score": source.original_score,
                    }
                    for source in row.sources
                ],
            }
            for row in result.documents
        ],
    }


def _decode(payload: bytes) -> TopicScoreResult:
    try:
        record = json.loads(payload)
        if _canonical(record) != payload:
            raise ValueError("topic-score encoding is not canonical")
        if record["schema_version"] != SCHEMA_VERSION:
            raise ValueError("unsupported topic-score schema")
        needs = tuple(
            InformationNeed(row["need_id"], row["text"]) for row in record["needs"]
        )
        documents = tuple(
            ScoredTopicDocument(
                document_id=row["document_id"],
                score=float(row["score"]),
                winning_need_id=row["winning_need_id"],
                passage_start=int(row["passage_start"]),
                passage_end=int(row["passage_end"]),
                passage_content_hash=bytes.fromhex(row["passage_content_hash"]),
                sources=tuple(
                    RetrievalSource(
                        task_id=source["task_id"],
                        query=source["query"],
                        object_key=source["object_key"],
                        original_rank=int(source["original_rank"]),
                        original_score=float(source["original_score"]),
                    )
                    for source in row["sources"]
                ),
            )
            for row in record["documents"]
        )
        result = TopicScoreResult(
            topic_id=record["topic_id"],
            representation=record["representation"],
            needs=needs,
            input_object_keys=tuple(record["input_object_keys"]),
            unique_document_count=int(record["unique_document_count"]),
            passage_count=int(record["passage_count"]),
            documents=documents,
            scorer_fingerprint=record["scorer_fingerprint"],
            chunker_fingerprint=record["chunker_fingerprint"],
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("invalid topic-score artifact") from error
    if result.unique_document_count != len(result.documents):
        raise ValueError("invalid topic-score document count")
    if any(not math.isfinite(row.score) for row in result.documents):
        raise ValueError("invalid topic-score value")
    if tuple(sorted(result.documents, key=lambda row: (-row.score, row.document_id))) != result.documents:
        raise ValueError("topic-score documents are not canonically ranked")
    return result


class TopicScoreArtifactCache:
    """Cache complete topic rankings under exact scorer and snapshot identities."""

    def __init__(self, store: object) -> None:
        if not callable(getattr(store, "read_validated", None)) or not callable(
            getattr(store, "write", None)
        ):
            raise TypeError("store must implement read_validated and write")
        self.store = store

    @staticmethod
    def object_key(fingerprint: TopicScoreFingerprint, request: TopicScoreRequest) -> str:
        return (
            f"{OBJECT_ROOT}/{fingerprint.digest}/"
            f"{request.digest[:2]}/{request.digest}.json"
        )

    def get(
        self, fingerprint: TopicScoreFingerprint, request: TopicScoreRequest
    ) -> TopicScoreResult | None:
        result = self.store.read_validated(self.object_key(fingerprint, request), _decode)
        if result is None:
            return None
        self._match(fingerprint, request, result)
        return result

    @staticmethod
    def _match(
        fingerprint: TopicScoreFingerprint,
        request: TopicScoreRequest,
        result: TopicScoreResult,
    ) -> None:
        if (
            result.topic_id != request.topic_id
            or result.needs != request.needs
            or result.input_object_keys != request.input_object_keys
            or result.representation != fingerprint.representation
            or result.scorer_fingerprint != fingerprint.scorer_fingerprint
            or result.chunker_fingerprint != fingerprint.chunker_fingerprint
        ):
            raise ValueError("topic-score result does not match request")

    def put(
        self,
        fingerprint: TopicScoreFingerprint,
        request: TopicScoreRequest,
        result: TopicScoreResult,
    ) -> bool:
        self._match(fingerprint, request, result)
        key = self.object_key(fingerprint, request)
        payload = _canonical(_result_record(result))
        if self.store.write(key, payload) is not True:
            return False
        evict = getattr(self.store, "evict_local", None)
        if callable(evict):
            evict(key)
        return self.get(fingerprint, request) == result


__all__ = [
    "TopicScoreArtifactCache",
    "TopicScoreFingerprint",
    "TopicScoreRequest",
]
