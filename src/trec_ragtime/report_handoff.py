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
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .retrieval_handoff import RetrievalHandoffBundle
from .retrieval_handoff_cache import (
    HandoffFingerprint,
    RetrievalHandoffArtifactCache,
)

HANDOFF_MANIFEST_SCHEMA_VERSION = 1


class HandoffSourceError(ValueError):
    """The handoff manifest or artifact cannot be consumed."""


@dataclass(frozen=True, slots=True)
class HandoffTopicEntry:
    """One topic row from the upstream handoff manifest."""

    topic_id: str
    status: str
    object_key: str | None
    producer_identity_digest: bytes | None = None

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
        producer = row.get("producer_identity_digest")
        if producer is not None:
            if not isinstance(producer, str):
                raise HandoffSourceError(
                    f"topic {topic_id} has an invalid producer_identity_digest"
                )
            try:
                producer = bytes.fromhex(producer)
            except ValueError:
                raise HandoffSourceError(
                    f"topic {topic_id} producer_identity_digest is not hex"
                ) from None
        topics.append(
            HandoffTopicEntry(topic_id, status, object_key or None, producer)
        )

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


def require_handoff_bundle(
    store: Any,
    entry: HandoffTopicEntry,
    *,
    topic_catalog_digest: bytes,
    fingerprint: HandoffFingerprint | None = None,
) -> RetrievalHandoffBundle:
    """Return the bundle for one manifest entry, failing closed if unusable.

    Reading goes through the upstream keyed reader, which recovers the request
    digest from the canonical object key and validates the complete request
    identity against the catalog and producer digests the manifest records.
    """

    if not entry.available:
        raise HandoffSourceError(
            f"topic {entry.topic_id} has no usable handoff artifact (status {entry.status})"
        )
    assert entry.object_key is not None
    if entry.producer_identity_digest is None:
        raise HandoffSourceError(
            f"topic {entry.topic_id} has no producer identity digest in the manifest"
        )
    cache = RetrievalHandoffArtifactCache(store)
    try:
        bundle = cache.get_by_object_key(
            fingerprint or HandoffFingerprint(),
            entry.object_key,
            topic_catalog_digest=topic_catalog_digest,
            producer_identity_digest=entry.producer_identity_digest,
        )
    except Exception as error:
        raise HandoffSourceError(
            f"handoff artifact for topic {entry.topic_id} is unreadable:"
            f" {type(error).__name__}: {error}"
        ) from None
    if bundle is None:
        reader = getattr(store, "read", None)
        present = callable(reader) and reader(entry.object_key) is not None
        detail = "is stored but invalid" if present else "is absent from the cache"
        raise HandoffSourceError(
            f"handoff artifact for topic {entry.topic_id} {detail}"
        )
    if bundle.topic.topic_id != entry.topic_id:
        raise HandoffSourceError(
            f"handoff artifact for topic {entry.topic_id} carries topic"
            f" {bundle.topic.topic_id}"
        )
    return bundle


def merge_handoff_manifests(paths: Sequence[Path | str]) -> HandoffManifest:
    """Combine several handoff manifests into one consumable index.

    A retry run writes its own manifest rather than rewriting the original, so
    the complete picture spans several files. Manifests are applied in the order
    given: a later written entry replaces an earlier one for the same topic, but
    a failure never replaces a success, because a retry manifest only reports on
    the topics it attempted.

    All manifests must share a fingerprint and topic catalog. Merging across
    different handoff semantics would silently mix incompatible evidence.
    """

    items = [load_handoff_manifest(path) for path in paths]
    if not items:
        raise HandoffSourceError("at least one handoff manifest is required")

    first = items[0]
    for other in items[1:]:
        if other.fingerprint != first.fingerprint:
            raise HandoffSourceError(
                f"handoff manifest {other.path_name} has fingerprint"
                f" {other.fingerprint[:12]}, expected {first.fingerprint[:12]}"
            )
        if other.topic_catalog_digest != first.topic_catalog_digest:
            raise HandoffSourceError(
                f"handoff manifest {other.path_name} was built against a"
                " different topic catalog"
            )

    merged: dict[str, HandoffTopicEntry] = {}
    for manifest in items:
        for entry in manifest.topics:
            existing = merged.get(entry.topic_id)
            if existing is not None and existing.available and not entry.available:
                continue
            merged[entry.topic_id] = entry

    topics = tuple(merged.values())
    return HandoffManifest(
        path_name=" + ".join(item.path_name for item in items),
        fingerprint=first.fingerprint,
        topic_catalog_digest=first.topic_catalog_digest,
        complete=all(entry.available for entry in topics),
        topics=topics,
    )


def catalog_digest_bytes(manifest: HandoffManifest) -> bytes:
    """Return the manifest's topic catalog digest as raw bytes."""

    try:
        return bytes.fromhex(manifest.topic_catalog_digest)
    except ValueError:
        raise HandoffSourceError("topic_catalog_digest is not hex") from None


__all__ = [
    "HANDOFF_MANIFEST_SCHEMA_VERSION",
    "catalog_digest_bytes",
    "HandoffManifest",
    "HandoffSourceError",
    "HandoffTopicEntry",
    "load_handoff_manifest",
    "merge_handoff_manifests",
    "require_handoff_bundle",
]
