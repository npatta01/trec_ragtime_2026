"""Parallel publication and manifest writing for generation handoffs."""

from __future__ import annotations

import dataclasses
import json
import os
import stat
import tempfile
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .retrieval_handoff import RetrievalHandoffBundle
from .retrieval_handoff_cache import (
    HandoffFingerprint,
    RetrievalHandoffArtifactCache,
    request_for_bundle,
)


DEFAULT_CONCURRENCY = 8


@dataclass(frozen=True, slots=True)
class HandoffWork:
    topic_id: str
    topic_catalog_digest: bytes
    bundle: RetrievalHandoffBundle


@dataclass(frozen=True, slots=True)
class HandoffOutcome:
    topic_id: str
    status: Literal["cached", "written", "failed"]
    object_key: str | None = None
    nugget_count: int = 0
    evidence_count: int = 0
    full_support_count: int = 0
    partial_support_count: int = 0
    citation_ready_count: int = 0
    reason: str | None = None


def run_topic(
    work: HandoffWork,
    *,
    fingerprint: HandoffFingerprint,
    cache: RetrievalHandoffArtifactCache,
) -> HandoffOutcome:
    """Publish or reuse one complete topic handoff."""

    try:
        if work.topic_id != work.bundle.topic.topic_id:
            raise ValueError("work topic identity disagrees with bundle")
        request = request_for_bundle(
            work.bundle, topic_catalog_digest=work.topic_catalog_digest
        )
        key = cache.object_key(fingerprint, request)
        cached = cache.get(fingerprint, request)
        if cached is not None:
            return _outcome(cached, "cached", key)
        durable = cache.put(fingerprint, request, work.bundle)
        return _outcome(durable, "written", key)
    except Exception as error:  # noqa: BLE001 - isolate one topic failure
        return HandoffOutcome(work.topic_id, "failed", reason=_sanitize(error))


def run_topics(
    work: Sequence[HandoffWork],
    *,
    fingerprint: HandoffFingerprint,
    cache: RetrievalHandoffArtifactCache,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> tuple[HandoffOutcome, ...]:
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
        raise ValueError("concurrency must be a positive integer")
    topic_ids = tuple(row.topic_id for row in work)
    if len(set(topic_ids)) != len(topic_ids):
        raise ValueError("handoff work topic IDs must be unique")
    if not work:
        return ()

    def one(row: HandoffWork) -> HandoffOutcome:
        return run_topic(row, fingerprint=fingerprint, cache=cache)

    with ThreadPoolExecutor(max_workers=min(concurrency, len(work))) as pool:
        return tuple(pool.map(one, work))


def write_handoff_manifest(
    path: Path,
    outcomes: Sequence[HandoffOutcome],
    *,
    topic_catalog_digest: bytes,
    fingerprint: HandoffFingerprint,
) -> None:
    if not isinstance(topic_catalog_digest, bytes) or len(topic_catalog_digest) != 32:
        raise ValueError("topic catalog digest must be 32 bytes")
    topic_ids = tuple(row.topic_id for row in outcomes)
    if len(set(topic_ids)) != len(topic_ids):
        raise ValueError("manifest topic IDs must be unique")
    payload = {
        "schema_version": 1,
        "fingerprint": fingerprint.digest.hex(),
        "topic_catalog_digest": topic_catalog_digest.hex(),
        "complete": bool(outcomes) and all(row.status != "failed" for row in outcomes),
        "topic_count": len(outcomes),
        "nugget_count": sum(row.nugget_count for row in outcomes),
        "evidence_count": sum(row.evidence_count for row in outcomes),
        "full_support_count": sum(row.full_support_count for row in outcomes),
        "partial_support_count": sum(row.partial_support_count for row in outcomes),
        "citation_ready_count": sum(row.citation_ready_count for row in outcomes),
        "topics": [dataclasses.asdict(row) for row in outcomes],
    }
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=".handoff-")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, stat.S_IRUSR | stat.S_IWUSR)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _outcome(
    bundle: RetrievalHandoffBundle,
    status: Literal["cached", "written"],
    key: str,
) -> HandoffOutcome:
    return HandoffOutcome(
        topic_id=bundle.topic.topic_id,
        status=status,
        object_key=key,
        nugget_count=bundle.counts.nuggets,
        evidence_count=bundle.counts.selected_evidence,
        full_support_count=bundle.counts.full_support,
        partial_support_count=bundle.counts.partial_support,
        citation_ready_count=bundle.counts.citation_ready,
    )


def _sanitize(error: BaseException) -> str:
    safe_types = {
        "ArtifactStoreRemoteError",
        "ArtifactValidationError",
        "HandoffError",
        "TypeError",
        "ValueError",
    }
    name = type(error).__name__
    return name if name in safe_types else "handoff topic failed"


__all__ = [
    "DEFAULT_CONCURRENCY",
    "HandoffOutcome",
    "HandoffWork",
    "run_topic",
    "run_topics",
    "write_handoff_manifest",
]
