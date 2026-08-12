"""One immutable nugget bundle per topic in the shared artifact store.

Identity is split the way ``retrieval_cache`` splits it. The fingerprint covers
slow-moving behaviour -- package revision, model, dedup and selection policy --
so a change there retires every old object. The request digest covers one
topic's exact inputs. Both appear in the object key, so an artifact can prove it
belongs where it was found.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable, Mapping, Sequence

import msgspec

from .artifact_store import (
    ArtifactStoreLike,
    ArtifactStoreRemoteError,
    ArtifactValidationError,
)
from .nuggets import NuggetBundle, NuggetError, NuggetEvidence, check_bundle


NUGGET_SCHEMA_VERSION = 1
NUGGET_OBJECT_KEY_PREFIX = "nuggets/v1"


class NuggetFingerprint(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Every result-affecting setting that is not specific to one topic."""

    schema_version: int
    behavior_version: str
    package: Mapping[str, str]
    nuggetizer_config: Mapping[str, object]
    provider_base_url_digest: bytes
    embedding_model: str
    embedding_revision: str
    similarity_threshold: float
    guard_version: str
    support_labels: tuple[str, ...]
    evidence_budget: int
    max_nuggets: int
    # Limits that decide what the model may emit and what survives the atomicity
    # guard. They live here so editing one retires previous artifacts instead of
    # silently changing what a cached bundle means.
    output_limits: Mapping[str, int]

    @property
    def digest(self) -> bytes:
        return hashlib.sha256(msgspec.msgpack.encode(self)).digest()


class NuggetInput(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """One retrieval artifact this bundle was built from."""

    facet_index: int
    query: str
    object_key: str
    request_digest: bytes


class NuggetRequest(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """The exact inputs that produce one topic's bundle.

    ``query`` is the topic question handed to both nugget creation and
    assignment, so it belongs in the identity: without it, rewording the
    question while the evidence stays fixed returns the previous bundle.

    ``evidence_digest`` pins the ordered alias table, so an artifact whose
    aliases, documents, or offsets were altered cannot resolve to this key.
    """

    topic_id: str
    query: str
    representation: str
    inputs: tuple[NuggetInput, ...]
    passage_source_fingerprint: str
    evidence_digest: bytes

    @property
    def digest(self) -> bytes:
        return hashlib.sha256(msgspec.msgpack.encode(self)).digest()


def evidence_digest(evidence: Sequence[NuggetEvidence]) -> bytes:
    """Digest the ordered evidence table exactly as it will be stored."""

    return hashlib.sha256(msgspec.msgpack.encode(tuple(evidence))).digest()


class NuggetEnvelope(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """The stored object, carrying the identity it was written under."""

    schema_version: int
    fingerprint: bytes
    request_digest: bytes
    stored_at: float
    bundle: NuggetBundle


def nugget_object_key(fingerprint: bytes, request_digest: bytes) -> str:
    """Build ``nuggets/v1/<fingerprint>/<prefix>/<request>.mpk``."""

    fingerprint_hex = _digest(fingerprint, name="nuggetizer fingerprint").hex()
    request_hex = _digest(request_digest, name="request digest").hex()
    return (
        f"{NUGGET_OBJECT_KEY_PREFIX}/{fingerprint_hex}/"
        f"{request_hex[:2]}/{request_hex}.mpk"
    )


class NuggetArtifactCache:
    """Typed cache adapter for one validated nugget bundle per identity."""

    def __init__(
        self, store: ArtifactStoreLike, *, clock: Callable[[], float] = time.time
    ) -> None:
        if not callable(getattr(store, "read_validated", None)) or not callable(
            getattr(store, "write", None)
        ):
            raise TypeError("store must implement read_validated and write")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.store = store
        self._clock = clock

    def object_key(self, fingerprint: NuggetFingerprint, request: NuggetRequest) -> str:
        return nugget_object_key(fingerprint.digest, request.digest)

    def get(
        self, fingerprint: NuggetFingerprint, request: NuggetRequest
    ) -> NuggetBundle | None:
        try:
            bundle = self.store.read_validated(
                self.object_key(fingerprint, request), _decoder(fingerprint, request)
            )
        except (OSError, ArtifactStoreRemoteError):
            return None
        if bundle is not None and not isinstance(bundle, NuggetBundle):
            raise ArtifactValidationError("cached nugget decoder returned an invalid value")
        return bundle

    def put(
        self,
        fingerprint: NuggetFingerprint,
        request: NuggetRequest,
        bundle: NuggetBundle,
    ) -> bool:
        """Store one complete bundle; an invalid bundle is never written."""

        _validate(fingerprint, request, bundle)
        stored_at = self._clock()
        if not isinstance(stored_at, (int, float)) or not math.isfinite(stored_at):
            raise ValueError("cache clock must return a finite number")
        envelope = NuggetEnvelope(
            schema_version=NUGGET_SCHEMA_VERSION,
            fingerprint=fingerprint.digest,
            request_digest=request.digest,
            stored_at=float(stored_at),
            bundle=bundle,
        )
        try:
            return (
                self.store.write(
                    self.object_key(fingerprint, request),
                    msgspec.msgpack.encode(envelope),
                )
                is True
            )
        except (OSError, ArtifactStoreRemoteError):
            return False


def _decoder(
    fingerprint: NuggetFingerprint, request: NuggetRequest
) -> Callable[[bytes], NuggetBundle]:
    def decode(payload: bytes) -> NuggetBundle:
        try:
            envelope = msgspec.msgpack.decode(payload, type=NuggetEnvelope)
        except (msgspec.DecodeError, msgspec.ValidationError, TypeError, ValueError):
            raise ArtifactValidationError("cached nugget envelope is invalid") from None
        try:
            if envelope.schema_version != NUGGET_SCHEMA_VERSION:
                raise ValueError("unsupported nugget schema version")
            if (
                envelope.fingerprint != fingerprint.digest
                or envelope.request_digest != request.digest
            ):
                raise ValueError("cached nugget identity does not match its key")
            if not math.isfinite(envelope.stored_at):
                raise ValueError("cached nugget timestamp is invalid")
            # A re-encode that differs means the bytes were not written by this
            # schema, so the object is treated as a miss and repaired.
            if msgspec.msgpack.encode(envelope) != payload:
                raise ValueError("cached nugget wire encoding is not canonical")
            _validate(fingerprint, request, envelope.bundle)
        except Exception:
            raise ArtifactValidationError("cached nugget payload is invalid") from None
        return envelope.bundle

    return decode


def _validate(
    fingerprint: NuggetFingerprint, request: NuggetRequest, bundle: NuggetBundle
) -> None:
    if not isinstance(bundle, NuggetBundle):
        raise TypeError("bundle must be a NuggetBundle")
    if bundle.topic_id != request.topic_id:
        raise NuggetError("nugget bundle names a different topic")
    if bundle.query != request.query:
        raise NuggetError("nugget bundle answers a different query")
    if bundle.representation != request.representation:
        raise NuggetError("nugget bundle uses a different source representation")
    if (
        bundle.evidence_budget != fingerprint.evidence_budget
        or bundle.max_nuggets != fingerprint.max_nuggets
    ):
        raise NuggetError("nugget bundle contradicts the selection fingerprint")
    # Pairs, not just indices: a caller that renumbers one side of the facet
    # mapping and not the other is caught here. A caller that renumbers both
    # consistently cannot be caught from inside, since facet_index is data the
    # caller supplies -- that invariant belongs to whoever holds the query plan.
    if {(facet.facet_index, facet.query) for facet in bundle.facets} != {
        (row.facet_index, row.query) for row in request.inputs
    }:
        raise NuggetError("nugget bundle covers a different facet set")
    # The request pins the alias table, so an altered alias, document, or offset
    # cannot be read back under this key.
    if evidence_digest(bundle.evidence) != request.evidence_digest:
        raise NuggetError("nugget evidence does not match the requested alias table")
    check_bundle(bundle)


def _digest(value: bytes | bytearray, *, name: str) -> bytes:
    if (
        not isinstance(value, (bytes, bytearray))
        or len(value) != hashlib.sha256().digest_size
    ):
        raise ValueError(f"{name} must be 32 SHA-256 bytes")
    return bytes(value)


__all__ = [
    "NUGGET_OBJECT_KEY_PREFIX",
    "NUGGET_SCHEMA_VERSION",
    "NuggetArtifactCache",
    "NuggetEnvelope",
    "NuggetFingerprint",
    "NuggetInput",
    "NuggetRequest",
    "evidence_digest",
    "nugget_object_key",
]
