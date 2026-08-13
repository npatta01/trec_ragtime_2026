"""Cache-only reading of retrieval handoff bundles for report generation.

Generation consumes exactly one self-contained bundle per topic and must not
reopen retrieval artifacts, nugget artifacts, the corpus, the Dataset Viewer, or
the organizer API. This module is the whole of that boundary: it resolves a
handoff object key from the upstream manifest and validates the artifact from
its own bytes.

The upstream ``RetrievalHandoffArtifactCache.get`` keys on a ``HandoffRequest``
whose ``selected_evidence_digest`` is derived from the bundle being fetched, so
a consumer that does not already hold the bundle cannot construct the key. The
manifest records the object key per topic, and ``HandoffEnvelope`` carries its
own fingerprint, request, and bundle digests, so reading by key is fully
self-validating. This performs no projection and no deduplication; it reuses
the upstream ``check_retrieval_handoff`` validator rather than restating it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import msgspec

from .artifact_store import ArtifactStoreLike, ArtifactValidationError
from .retrieval_handoff import (
    RetrievalHandoffBundle,
    check_retrieval_handoff,
    handoff_bundle_digest,
)
from .retrieval_handoff_cache import HandoffEnvelope, handoff_object_key

HANDOFF_MANIFEST_SCHEMA_VERSION = 1


class HandoffSourceError(ValueError):
    """The handoff manifest or artifact cannot be consumed."""


@dataclass(frozen=True, slots=True)
class HandoffTopicEntry:
    """One topic row from the upstream handoff manifest."""

    topic_id: str
    status: str
    object_key: str | None

    @property
    def available(self) -> bool:
        return self.status != "failed" and bool(self.object_key)


@dataclass(frozen=True, slots=True)
class HandoffManifest:
    """The upstream handoff run's index of per-topic artifacts."""

    path_name: str
    fingerprint: str
    topic_catalog_digest: str
    complete: bool
    topics: tuple[HandoffTopicEntry, ...]

    def entry(self, topic_id: str) -> HandoffTopicEntry | None:
        for row in self.topics:
            if row.topic_id == topic_id:
                return row
        return None

    @property
    def available_topic_ids(self) -> tuple[str, ...]:
        return tuple(row.topic_id for row in self.topics if row.available)


def load_handoff_manifest(path: Path | str) -> HandoffManifest:
    """Read the manifest written by ``ragtime-retrieval-handoff``."""

    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise HandoffSourceError(f"could not read handoff manifest: {error}") from None
    except json.JSONDecodeError as error:
        raise HandoffSourceError(f"handoff manifest is not valid JSON: {error}") from None
    if not isinstance(payload, dict):
        raise HandoffSourceError("handoff manifest must be a JSON object")
    if payload.get("schema_version") != HANDOFF_MANIFEST_SCHEMA_VERSION:
        raise HandoffSourceError(
            f"unsupported handoff manifest schema {payload.get('schema_version')!r}"
        )

    rows = payload.get("topics")
    if not isinstance(rows, list) or not rows:
        raise HandoffSourceError("handoff manifest lists no topics")
    topics: list[HandoffTopicEntry] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise HandoffSourceError("every handoff manifest topic must be an object")
        topic_id = row.get("topic_id")
        status = row.get("status")
        object_key = row.get("object_key")
        if not isinstance(topic_id, str) or not topic_id.strip():
            raise HandoffSourceError("every handoff manifest topic needs a topic_id")
        if topic_id in seen:
            raise HandoffSourceError(f"handoff manifest repeats topic {topic_id}")
        seen.add(topic_id)
        if not isinstance(status, str) or not status.strip():
            raise HandoffSourceError(f"topic {topic_id} has no status")
        if object_key is not None and not isinstance(object_key, str):
            raise HandoffSourceError(f"topic {topic_id} has an invalid object_key")
        topics.append(HandoffTopicEntry(topic_id, status, object_key or None))

    fingerprint = payload.get("fingerprint")
    catalog = payload.get("topic_catalog_digest")
    for value, name in ((fingerprint, "fingerprint"), (catalog, "topic_catalog_digest")):
        if not isinstance(value, str) or not value.strip():
            raise HandoffSourceError(f"handoff manifest {name} is missing")

    return HandoffManifest(
        path_name=path.name,
        fingerprint=fingerprint,
        topic_catalog_digest=catalog,
        complete=bool(payload.get("complete")),
        topics=tuple(topics),
    )


def _decoder(object_key: str):
    def decode(payload: bytes) -> RetrievalHandoffBundle:
        try:
            envelope = msgspec.msgpack.decode(payload, type=HandoffEnvelope)
            if msgspec.msgpack.encode(envelope) != payload:
                raise ValueError("noncanonical encoding")
            if envelope.bundle_digest != handoff_bundle_digest(envelope.bundle):
                raise ValueError("bundle digest mismatch")
            expected = handoff_object_key(
                envelope.fingerprint_digest, envelope.request_digest
            )
            if expected != object_key:
                raise ValueError("object key does not match the stored identity")
            check_retrieval_handoff(envelope.bundle)
        except (msgspec.DecodeError, msgspec.ValidationError, TypeError, ValueError):
            raise ArtifactValidationError("cached handoff payload is invalid") from None
        return envelope.bundle

    return decode


def read_handoff_bundle(
    store: ArtifactStoreLike, object_key: str
) -> RetrievalHandoffBundle | None:
    """Return a validated handoff bundle, or ``None`` when it is genuinely absent."""

    if not isinstance(object_key, str) or not object_key:
        raise HandoffSourceError("handoff object key must be a non-empty string")
    bundle = store.read_validated(object_key, _decoder(object_key))
    if bundle is None:
        return None
    if not isinstance(bundle, RetrievalHandoffBundle):
        raise ArtifactValidationError("handoff decoder returned an invalid value")
    return bundle


def require_handoff_bundle(
    store: ArtifactStoreLike, entry: HandoffTopicEntry
) -> RetrievalHandoffBundle:
    """Return the bundle for one manifest entry, failing closed if unusable."""

    if not entry.available:
        raise HandoffSourceError(
            f"topic {entry.topic_id} has no usable handoff artifact (status {entry.status})"
        )
    assert entry.object_key is not None
    bundle = read_handoff_bundle(store, entry.object_key)
    if bundle is None:
        raise HandoffSourceError(
            f"handoff artifact for topic {entry.topic_id} is absent from the cache"
        )
    if bundle.topic.topic_id != entry.topic_id:
        raise HandoffSourceError(
            f"handoff artifact for topic {entry.topic_id} carries topic"
            f" {bundle.topic.topic_id}"
        )
    return bundle


__all__ = [
    "HANDOFF_MANIFEST_SCHEMA_VERSION",
    "HandoffManifest",
    "HandoffSourceError",
    "HandoffTopicEntry",
    "load_handoff_manifest",
    "read_handoff_bundle",
    "require_handoff_bundle",
]
