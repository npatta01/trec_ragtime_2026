"""Typed durable cache for retrieval-to-generation handoff bundles."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Literal

import msgspec

from .artifact_store import ArtifactStoreLike, ArtifactStoreRemoteError, ArtifactValidationError
from .retrieval_handoff import (
    HandoffNuggetSource,
    HandoffRetrievalInput,
    HandoffTopic,
    RetrievalHandoffBundle,
    check_retrieval_handoff,
    handoff_bundle_digest,
)


OBJECT_ROOT = "retrieval-handoff/v1"


class HandoffFingerprint(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    schema_version: Literal[1] = 1
    behavior_version: str = "retrieval-handoff-v1"
    selection_policy: str = "retained-support-alias-union-v1"
    citation_policy: str = "full-support-only-v1"
    representation: Literal["report_text"] = "report_text"
    validation_version: str = "strict-source-slice-v1"

    @property
    def digest(self) -> bytes:
        return hashlib.sha256(msgspec.msgpack.encode(self)).digest()


class HandoffRequest(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    topic_catalog_digest: bytes
    topic: HandoffTopic
    nugget_source: HandoffNuggetSource
    retrieval_inputs: tuple[HandoffRetrievalInput, ...]
    selected_evidence_digest: bytes

    @property
    def digest(self) -> bytes:
        return hashlib.sha256(msgspec.msgpack.encode(self)).digest()


class HandoffEnvelope(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    fingerprint_digest: bytes
    request_digest: bytes
    bundle_digest: bytes
    bundle: RetrievalHandoffBundle


def selected_evidence_digest(bundle: RetrievalHandoffBundle) -> bytes:
    check_retrieval_handoff(bundle)
    return hashlib.sha256(msgspec.msgpack.encode(bundle.evidence)).digest()


def request_for_bundle(
    bundle: RetrievalHandoffBundle, *, topic_catalog_digest: bytes
) -> HandoffRequest:
    return HandoffRequest(
        topic_catalog_digest=_digest(topic_catalog_digest, "topic catalog"),
        topic=bundle.topic,
        nugget_source=bundle.nugget_source,
        retrieval_inputs=bundle.retrieval_inputs,
        selected_evidence_digest=selected_evidence_digest(bundle),
    )


def handoff_object_key(fingerprint: bytes, request_digest: bytes) -> str:
    fingerprint_hex = _digest(fingerprint, "handoff fingerprint").hex()
    request_hex = _digest(request_digest, "handoff request").hex()
    return f"{OBJECT_ROOT}/{fingerprint_hex}/{request_hex[:2]}/{request_hex}.mpk"


class RetrievalHandoffArtifactCache:
    """Read and publish only validated, self-contained handoff bundles."""

    def __init__(self, store: ArtifactStoreLike) -> None:
        for method in ("read_validated", "write", "evict_local"):
            if not callable(getattr(store, method, None)):
                raise TypeError(f"store must implement {method}")
        self.store = store

    def object_key(self, fingerprint: HandoffFingerprint, request: HandoffRequest) -> str:
        return handoff_object_key(fingerprint.digest, request.digest)

    def get(
        self, fingerprint: HandoffFingerprint, request: HandoffRequest
    ) -> RetrievalHandoffBundle | None:
        bundle = self.store.read_validated(
            self.object_key(fingerprint, request), _decoder(fingerprint, request)
        )
        if bundle is not None and not isinstance(bundle, RetrievalHandoffBundle):
            raise ArtifactValidationError("handoff decoder returned an invalid value")
        return bundle

    def put(
        self,
        fingerprint: HandoffFingerprint,
        request: HandoffRequest,
        bundle: RetrievalHandoffBundle,
    ) -> RetrievalHandoffBundle:
        _validate(fingerprint, request, bundle)
        envelope = HandoffEnvelope(
            fingerprint_digest=fingerprint.digest,
            request_digest=request.digest,
            bundle_digest=handoff_bundle_digest(bundle),
            bundle=bundle,
        )
        key = self.object_key(fingerprint, request)
        if self.store.write(key, msgspec.msgpack.encode(envelope)) is not True:
            raise ArtifactStoreRemoteError("handoff artifact write did not succeed")
        self.store.evict_local(key)
        durable = self.get(fingerprint, request)
        if durable is None:
            raise ArtifactStoreRemoteError("handoff artifact durable reread failed")
        return durable


def _decoder(
    fingerprint: HandoffFingerprint, request: HandoffRequest
) -> Callable[[bytes], RetrievalHandoffBundle]:
    def decode(payload: bytes) -> RetrievalHandoffBundle:
        try:
            envelope = msgspec.msgpack.decode(payload, type=HandoffEnvelope)
            if msgspec.msgpack.encode(envelope) != payload:
                raise ValueError("noncanonical encoding")
            if (
                envelope.fingerprint_digest != fingerprint.digest
                or envelope.request_digest != request.digest
                or envelope.bundle_digest != handoff_bundle_digest(envelope.bundle)
            ):
                raise ValueError("identity mismatch")
            _validate(fingerprint, request, envelope.bundle)
        except (msgspec.DecodeError, msgspec.ValidationError, TypeError, ValueError):
            raise ArtifactValidationError("cached handoff payload is invalid") from None
        return envelope.bundle

    return decode


def _validate(
    fingerprint: HandoffFingerprint,
    request: HandoffRequest,
    bundle: RetrievalHandoffBundle,
) -> None:
    if not isinstance(fingerprint, HandoffFingerprint):
        raise TypeError("fingerprint must be a HandoffFingerprint")
    for value, name in (
        (request.topic_catalog_digest, "topic catalog"),
        (request.selected_evidence_digest, "selected evidence"),
    ):
        _digest(value, name)
    check_retrieval_handoff(bundle)
    if (
        request.topic != bundle.topic
        or request.nugget_source != bundle.nugget_source
        or request.retrieval_inputs != bundle.retrieval_inputs
        or request.selected_evidence_digest != selected_evidence_digest(bundle)
    ):
        raise ValueError("handoff bundle does not match its request")


def _digest(value: object, name: str) -> bytes:
    if not isinstance(value, bytes) or len(value) != 32:
        raise ValueError(f"{name} digest must be 32 bytes")
    return value


__all__ = [
    "HandoffEnvelope",
    "HandoffFingerprint",
    "HandoffRequest",
    "RetrievalHandoffArtifactCache",
    "handoff_object_key",
    "request_for_bundle",
    "selected_evidence_digest",
]
