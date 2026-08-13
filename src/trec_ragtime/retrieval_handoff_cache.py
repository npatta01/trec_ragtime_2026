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
        for method in ("read_validated_strict", "write", "evict_local"):
            if not callable(getattr(store, method, None)):
                raise TypeError(f"store must implement {method}")
        self.store = store

    def object_key(self, fingerprint: HandoffFingerprint, request: HandoffRequest) -> str:
        return handoff_object_key(fingerprint.digest, request.digest)

    def get(
        self, fingerprint: HandoffFingerprint, request: HandoffRequest
    ) -> RetrievalHandoffBundle | None:
        bundle = self.store.read_validated_strict(
            self.object_key(fingerprint, request), _decoder(fingerprint, request)
        )
        if bundle is not None and not isinstance(bundle, RetrievalHandoffBundle):
            raise ArtifactValidationError("handoff decoder returned an invalid value")
        return bundle

    def get_by_object_key(
        self,
        fingerprint: HandoffFingerprint,
        object_key: str,
        *,
        topic_catalog_digest: bytes,
    ) -> RetrievalHandoffBundle | None:
        """Read one listed handoff without rebuilding its upstream request.

        The request digest is recoverable from the canonical object key, while
        the rest of the request is recoverable from the envelope's bundle.  A
        catalog digest supplied by the caller is therefore enough to validate
        the complete request identity without loading topics, retrievals, or
        nugget artifacts.
        """

        request_digest = _request_digest_from_object_key(fingerprint, object_key)
        try:
            catalog_digest = _digest(topic_catalog_digest, "topic catalog")
        except (TypeError, ValueError):
            raise ArtifactValidationError("topic catalog digest is invalid") from None
        bundle = self.store.read_validated_strict(
            object_key,
            _object_key_decoder(
                fingerprint,
                request_digest,
                topic_catalog_digest=catalog_digest,
            ),
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
            envelope = _decode_envelope(payload)
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


def _object_key_decoder(
    fingerprint: HandoffFingerprint,
    request_digest: bytes,
    *,
    topic_catalog_digest: bytes,
) -> Callable[[bytes], RetrievalHandoffBundle | None]:
    def decode(payload: bytes) -> RetrievalHandoffBundle | None:
        try:
            envelope = _decode_envelope(payload)
            if envelope.fingerprint_digest != fingerprint.digest:
                raise ValueError("fingerprint identity mismatch")
            if envelope.request_digest != request_digest:
                raise ValueError("request identity does not match object key")
            if envelope.bundle_digest != handoff_bundle_digest(envelope.bundle):
                raise ValueError("bundle identity mismatch")
            request = request_for_bundle(
                envelope.bundle,
                topic_catalog_digest=topic_catalog_digest,
            )
            if envelope.request_digest != request.digest:
                # A valid object under this fingerprint can belong to another
                # topic-catalog snapshot.  It is a confirmed identity miss,
                # not malformed data, so callers may continue discovery.
                return None
            _validate(fingerprint, request, envelope.bundle)
        except Exception:
            raise ArtifactValidationError("cached handoff payload is invalid") from None
        return envelope.bundle

    return decode


def _decode_envelope(payload: bytes) -> HandoffEnvelope:
    try:
        envelope = msgspec.msgpack.decode(payload, type=HandoffEnvelope)
        if msgspec.msgpack.encode(envelope) != payload:
            raise ValueError("noncanonical encoding")
    except Exception:
        raise ArtifactValidationError("cached handoff envelope is invalid") from None
    return envelope


def _request_digest_from_object_key(
    fingerprint: HandoffFingerprint, object_key: str
) -> bytes:
    if not isinstance(fingerprint, HandoffFingerprint):
        raise TypeError("fingerprint must be a HandoffFingerprint")
    if not isinstance(object_key, str):
        raise ArtifactValidationError("handoff object key is invalid")
    parts = object_key.split("/")
    root_parts = OBJECT_ROOT.split("/")
    if len(parts) != len(root_parts) + 3 or tuple(parts[: len(root_parts)]) != tuple(root_parts):
        raise ArtifactValidationError("handoff object key is invalid")
    if parts[len(root_parts)] != fingerprint.digest.hex():
        raise ArtifactValidationError("handoff object key is outside the fingerprint root")

    shard = parts[-2]
    request_name = parts[-1]
    if not request_name.endswith(".mpk"):
        raise ArtifactValidationError("handoff object key is invalid")
    request_hex = request_name[: -len(".mpk")]
    if (
        len(request_hex) != hashlib.sha256().digest_size * 2
        or request_hex.lower() != request_hex
        or any(character not in "0123456789abcdef" for character in request_hex)
        or shard != request_hex[:2]
    ):
        raise ArtifactValidationError("handoff object key is invalid")
    return bytes.fromhex(request_hex)


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
