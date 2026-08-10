"""Strict, immutable artifacts for complete hydrated retrievals.

The retrieval cache deliberately sits above :mod:`artifact_store`: this module
owns the semantic identity and the typed MessagePack payload, while
``ArtifactStore`` owns L1/L2 lookup, repair, and transport failures.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

import msgspec

from .artifact_store import ArtifactStoreLike, ArtifactValidationError
from .cache import _normalize_base_url, canonicalize_json
from .models import (
    HydrationFailure,
    PipelineResult,
    RetrievalResult,
    RetrievedDocument,
    SearchHit,
)

RETRIEVAL_SCHEMA_VERSION = 1
RETRIEVAL_OBJECT_KEY_PREFIX = "retrieval/v1"

# These versions are intentionally explicit.  A change in ranking, hydration,
# fallback, translation, or endpoint-depth behavior must invalidate old
# retrieval objects even when the query itself is unchanged.
CLIENT_BEHAVIOR_VERSION = "client-v1"
RETRIEVER_BEHAVIOR_VERSION = "retriever-v1"
ENDPOINT_DEPTH_BEHAVIOR_VERSION = "endpoint-depth-v1"
HYDRATION_BEHAVIOR_VERSION = "hydration-v1"
FALLBACK_BEHAVIOR_VERSION = "fallback-v1"
TRANSLATION_BEHAVIOR_VERSION = "translation-v1"

def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


class FrozenDict(dict[str, Any]):
    """A JSON/MessagePack/pickle-compatible recursively immutable mapping.

    Normal instance mutators, including re-initialization, are rejected after
    construction. Explicit base-class calls such as ``dict.__setitem__`` or
    ``dict.update`` remain Python subclass escape hatches by design.
    """

    __slots__ = ("_sealed",)

    def __new__(cls, *args: Any, **kwargs: Any) -> "FrozenDict":
        instance = dict.__new__(cls)
        object.__setattr__(instance, "_sealed", False)
        return instance

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if self._sealed:
            raise TypeError("frozen identity mappings cannot be reinitialized")
        # Bypass the blocked public mutators while constructing the value.
        dict.__init__(self, *args, **kwargs)
        object.__setattr__(self, "_sealed", True)

    @staticmethod
    def _immutable(*args: Any, **kwargs: Any) -> None:
        raise TypeError("frozen identity mappings cannot be mutated")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable

    def __ior__(self, other: Any) -> "FrozenDict":
        raise TypeError("frozen identity mappings cannot be mutated")

    @classmethod
    def _from_mapping(cls, value: Mapping[str, Any]) -> "FrozenDict":
        """Private deserialization seam that seals without public ``__init__``."""

        instance = cls.__new__(cls)
        dict.__init__(instance, value)
        object.__setattr__(instance, "_sealed", True)
        return instance

    def __reduce__(self) -> tuple[Any, tuple[dict[str, Any]]]:
        return (FrozenDict._from_mapping, (dict(self),))

    def __reduce_ex__(self, protocol: int) -> tuple[Any, tuple[dict[str, Any]]]:
        return self.__reduce__()


def _canonical_json(value: Any) -> Any:
    """Canonicalize mappings while accepting immutable tuple/list fixtures."""

    if isinstance(value, tuple):
        return [_canonical_json(item) for item in value]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("cache mappings must have string keys")
        return canonicalize_json({key: _canonical_json(nested) for key, nested in value.items()})
    return canonicalize_json(value)


def _freeze_json(value: Any) -> Any:
    """Return a recursively immutable canonical JSON-like value."""

    canonical = _canonical_json(value)
    if isinstance(canonical, dict):
        return FrozenDict(
            {key: _freeze_json(nested) for key, nested in canonical.items()}
        )
    if isinstance(canonical, list):
        return tuple(_freeze_json(item) for item in canonical)
    return canonical


def _corpus_without_path(corpus_identity: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(corpus_identity, Mapping):
        raise TypeError("corpus_identity must be a mapping")
    value = _canonical_json({key: nested for key, nested in corpus_identity.items() if key != "path"})
    if not isinstance(value, dict):
        raise ValueError("corpus_identity must be an object")
    return value


def _digest(value: bytes | bytearray | str, *, name: str) -> bytes:
    if isinstance(value, str):
        try:
            value = bytes.fromhex(value)
        except ValueError:
            raise ValueError(f"{name} must be 32 SHA-256 bytes") from None
    if not isinstance(value, (bytes, bytearray)) or len(value) != hashlib.sha256().digest_size:
        raise ValueError(f"{name} must be 32 SHA-256 bytes")
    return bytes(value)


def _validate_version(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _validate_schema_version(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value != RETRIEVAL_SCHEMA_VERSION:
        raise ValueError("unsupported retrieval schema version")
    return value


def retriever_fingerprint(
    base_url: str | RetrievalRequestIdentity,
    corpus_identity: Mapping[str, Any] | None = None,
    *,
    schema_version: int = RETRIEVAL_SCHEMA_VERSION,
    client_behavior_version: str = CLIENT_BEHAVIOR_VERSION,
    retriever_behavior_version: str = RETRIEVER_BEHAVIOR_VERSION,
    hydration_behavior_version: str = HYDRATION_BEHAVIOR_VERSION,
    fallback_behavior_version: str = FALLBACK_BEHAVIOR_VERSION,
    translation_behavior_version: str = TRANSLATION_BEHAVIOR_VERSION,
) -> bytes:
    """Return the binary fingerprint for retriever and corpus semantics."""

    if isinstance(base_url, RetrievalRequestIdentity):
        if corpus_identity is not None:
            raise TypeError("corpus_identity is not accepted with a request identity")
        request = base_url
        return request.retriever_fingerprint
    if corpus_identity is None:
        raise TypeError("corpus_identity is required")
    _validate_schema_version(schema_version)
    normalized = _normalize_base_url(base_url)
    payload = {
        "schema_version": schema_version,
        "base_url_digest": hashlib.sha256(normalized.encode("utf-8")).digest(),
        "client_behavior_version": _validate_version(
            client_behavior_version, name="client_behavior_version"
        ),
        "retriever_behavior_version": _validate_version(
            retriever_behavior_version, name="retriever_behavior_version"
        ),
        "hydration_behavior_version": _validate_version(
            hydration_behavior_version, name="hydration_behavior_version"
        ),
        "fallback_behavior_version": _validate_version(
            fallback_behavior_version, name="fallback_behavior_version"
        ),
        "translation_behavior_version": _validate_version(
            translation_behavior_version, name="translation_behavior_version"
        ),
        "corpus_identity": _corpus_without_path(corpus_identity),
    }
    return hashlib.sha256(msgspec.msgpack.encode(payload)).digest()


@dataclass(frozen=True, slots=True)
class RetrievalRequestIdentity:
    """The exact semantic inputs that produce one hydrated retrieval."""

    query: str
    pipeline: str
    top_k: int | None
    endpoint_depth_version: str
    max_content_fallbacks: int | None
    fallback_on_hf_error: bool
    content_collection: str
    backend_policy: Any
    query_context: Mapping[str, Any] | None
    base_url: str
    corpus_identity: Mapping[str, Any]
    schema_version: int = RETRIEVAL_SCHEMA_VERSION
    client_behavior_version: str = CLIENT_BEHAVIOR_VERSION
    retriever_behavior_version: str = RETRIEVER_BEHAVIOR_VERSION
    hydration_behavior_version: str = HYDRATION_BEHAVIOR_VERSION
    fallback_behavior_version: str = FALLBACK_BEHAVIOR_VERSION
    translation_behavior_version: str = TRANSLATION_BEHAVIOR_VERSION

    def __post_init__(self) -> None:
        _validate_schema_version(self.schema_version)
        if not isinstance(self.query, str):
            raise TypeError("query must be a string")
        if not isinstance(self.pipeline, str) or not self.pipeline:
            raise ValueError("pipeline must be a non-empty string")
        if self.top_k is not None and (
            isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 1
        ):
            raise ValueError("top_k must be a positive integer or None")
        _validate_version(self.endpoint_depth_version, name="endpoint_depth_version")
        if self.max_content_fallbacks is not None and (
            isinstance(self.max_content_fallbacks, bool)
            or not isinstance(self.max_content_fallbacks, int)
            or self.max_content_fallbacks < 0
        ):
            raise ValueError("max_content_fallbacks must be a non-negative integer or None")
        if not isinstance(self.fallback_on_hf_error, bool):
            raise TypeError("fallback_on_hf_error must be a boolean")
        if self.fallback_on_hf_error and self.max_content_fallbacks is None:
            raise ValueError("fallback_on_hf_error requires finite max_content_fallbacks")
        if not isinstance(self.content_collection, str) or not self.content_collection:
            raise ValueError("content_collection must be a non-empty string")
        _normalize_base_url(self.base_url)
        if self.query_context is not None and not isinstance(self.query_context, Mapping):
            raise TypeError("query_context must be a mapping or None")
        object.__setattr__(self, "base_url", _normalize_base_url(self.base_url))
        object.__setattr__(self, "backend_policy", _freeze_json(self.backend_policy))
        object.__setattr__(
            self,
            "query_context",
            None if self.query_context is None else _freeze_json(self.query_context),
        )
        object.__setattr__(
            self,
            "corpus_identity",
            _freeze_json(_corpus_without_path(self.corpus_identity)),
        )
        for name in (
            "client_behavior_version",
            "retriever_behavior_version",
            "hydration_behavior_version",
            "fallback_behavior_version",
            "translation_behavior_version",
        ):
            _validate_version(getattr(self, name), name=name)

    @property
    def retriever_fingerprint(self) -> bytes:
        return retriever_fingerprint(
            self.base_url,
            self.corpus_identity,
            schema_version=self.schema_version,
            client_behavior_version=self.client_behavior_version,
            retriever_behavior_version=self.retriever_behavior_version,
            hydration_behavior_version=self.hydration_behavior_version,
            fallback_behavior_version=self.fallback_behavior_version,
            translation_behavior_version=self.translation_behavior_version,
        )

    @property
    def canonical_request_bytes(self) -> bytes:
        payload = {
            "schema_version": self.schema_version,
            "base_url_digest": hashlib.sha256(self.base_url.encode("utf-8")).digest(),
            # Bytes preserve the exact UTF-8 query representation in the hash.
            "query_utf8": self.query.encode("utf-8"),
            "pipeline": self.pipeline,
            "top_k": self.top_k,
            "endpoint_depth_version": self.endpoint_depth_version,
            "max_content_fallbacks": self.max_content_fallbacks,
            "fallback_on_hf_error": self.fallback_on_hf_error,
            "content_collection": self.content_collection,
            "backend_policy": _canonical_json(self.backend_policy),
            "query_context": (
                None if self.query_context is None else _canonical_json(self.query_context)
            ),
        }
        return msgspec.msgpack.encode(payload)

    @property
    def request_digest(self) -> bytes:
        return hashlib.sha256(self.canonical_request_bytes).digest()

    @property
    def base_url_digest(self) -> bytes:
        return hashlib.sha256(self.base_url.encode("utf-8")).digest()

    @property
    def request_digest_hex(self) -> str:
        return self.request_digest.hex()

    @property
    def retriever_fingerprint_hex(self) -> str:
        return self.retriever_fingerprint.hex()

    @property
    def object_key(self) -> str:
        return retrieval_object_key(self.retriever_fingerprint, self.request_digest)


def retrieval_object_key(
    fingerprint: RetrievalRequestIdentity | bytes | bytearray | str,
    request_digest: bytes | bytearray | str | None = None,
) -> str:
    """Build ``retrieval/v1/<fingerprint>/<prefix>/<request>.mpk``."""

    if isinstance(fingerprint, RetrievalRequestIdentity):
        if request_digest is not None:
            raise TypeError("request_digest is not accepted with a request identity")
        request = fingerprint
        fingerprint_bytes = request.retriever_fingerprint
        request_bytes = request.request_digest
    else:
        fingerprint_bytes = _digest(fingerprint, name="retriever fingerprint")
        request_bytes = _digest(request_digest, name="request digest")
    fingerprint_hex = fingerprint_bytes.hex()
    request_hex = request_bytes.hex()
    return (
        f"{RETRIEVAL_OBJECT_KEY_PREFIX}/{fingerprint_hex}/"
        f"{request_hex[:2]}/{request_hex}.mpk"
    )


@dataclass(frozen=True, slots=True)
class RetrievalArtifactReference:
    """Small immutable provenance record safe for runner checkpoints."""

    object_key: str
    request_digest: bytes
    retriever_fingerprint: bytes
    stored_at: float
    cache_hit: bool

    def __post_init__(self) -> None:
        request_digest = _digest(self.request_digest, name="request digest")
        fingerprint = _digest(self.retriever_fingerprint, name="retriever fingerprint")
        _finite(self.stored_at, name="stored_at")
        if not isinstance(self.cache_hit, bool):
            raise TypeError("cache_hit must be a boolean")
        if not isinstance(self.object_key, str) or not self.object_key:
            raise ValueError("object_key must be a non-empty string")
        if self.object_key != retrieval_object_key(fingerprint, request_digest):
            raise ValueError("object_key does not match retrieval identity")

    @property
    def request_digest_hex(self) -> str:
        return self.request_digest.hex()

    @property
    def retriever_fingerprint_hex(self) -> str:
        return self.retriever_fingerprint.hex()

    def as_record(self) -> dict[str, object]:
        return {
            "object_key": self.object_key,
            "request_digest": self.request_digest_hex,
            "retriever_fingerprint": self.retriever_fingerprint_hex,
            "stored_at": float(self.stored_at),
            "cache_hit": self.cache_hit,
        }


class RetrievalHitWire(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    rank: int
    document_id: str
    score: float


class RetrievalPipelineWire(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    query: str
    pipeline: str
    hits: tuple[RetrievalHitWire, ...]
    processed: bool | None
    local_cache_hit: bool
    remote_cached: bool | None
    server_timestamp: float | None
    cache_stored_at: float | None
    extra: dict[str, Any]


class RetrievalDocumentWire(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    rank: int
    document_id: str
    score: float
    text: str | None
    url: str | None
    date: str | None
    language: str | None
    content_source: str | None
    content_local_cache_hit: bool | None
    english_text: str | None


class RetrievalFailureWire(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    split: str
    message: str


class RetrievalResultWire(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    search_result: RetrievalPipelineWire
    documents: tuple[RetrievalDocumentWire, ...]
    hydration_failures: tuple[RetrievalFailureWire, ...]
    missing_translation_ids: tuple[str, ...]


class RetrievalEnvelope(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    schema_version: int
    retriever_fingerprint: bytes
    request_digest: bytes
    stored_at: float
    result: RetrievalResultWire

    def __post_init__(self) -> None:
        _validate_schema_version(self.schema_version)


_SUPPORTED_LANGUAGES = frozenset({"eng", "spa", "rus", "zho"})
_CORPUS_CONTENT_SOURCES = frozenset(
    {"huggingface", "huggingface_local_index", "local_corpus"}
)


def _validate_result(request: RetrievalRequestIdentity, result: RetrievalResult) -> None:
    if not isinstance(result, RetrievalResult):
        raise TypeError("retrieval result must be a RetrievalResult")
    search = result.search_result
    if not isinstance(search, PipelineResult):
        raise TypeError("retrieval result search_result is invalid")
    if search.query != request.query or search.pipeline != request.pipeline:
        raise ValueError("retrieval result does not match request query or pipeline")
    if not isinstance(search.processed, (bool, type(None))):
        raise TypeError("processed must be a boolean or None")
    if not isinstance(search.local_cache_hit, bool):
        raise TypeError("local_cache_hit must be a boolean")
    for name, value in (
        ("remote_cached", search.remote_cached),
        ("server_timestamp", search.server_timestamp),
        ("cache_stored_at", search.cache_stored_at),
    ):
        if name == "remote_cached" and value is not None and not isinstance(value, bool):
            raise TypeError(f"{name} must be a boolean or None")
        if name != "remote_cached" and value is not None:
            _finite(value, name=name)
    _canonical_json(search.extra)

    hits = tuple(search.hits)
    expected_ids: set[str] = set()
    for expected_rank, hit in enumerate(hits, start=1):
        if not isinstance(hit, SearchHit):
            raise TypeError("search hits must be SearchHit values")
        if isinstance(hit.rank, bool) or not isinstance(hit.rank, int) or hit.rank != expected_rank:
            raise ValueError("search hit ranks must be contiguous and one-based")
        if not isinstance(hit.document_id, str) or not hit.document_id:
            raise ValueError("search hit IDs must be non-empty strings")
        if hit.document_id in expected_ids:
            raise ValueError("search hit IDs must be unique")
        expected_ids.add(hit.document_id)
        _finite(hit.score, name="search score")

    documents = tuple(result.documents)
    if len(documents) != len(hits):
        raise ValueError("hydrated documents must align one-to-one with search hits")
    documents_by_id: dict[str, RetrievedDocument] = {}
    for hit, document in zip(hits, documents, strict=True):
        if not isinstance(document, RetrievedDocument):
            raise TypeError("documents must be RetrievedDocument values")
        if isinstance(document.rank, bool) or not isinstance(document.rank, int):
            raise TypeError("document rank must be an integer")
        if (
            document.rank != hit.rank
            or document.document_id != hit.document_id
            or document.score != hit.score
        ):
            raise ValueError("hydrated documents must preserve hit rank, ID, and score")
        if document.document_id in documents_by_id:
            raise ValueError("document IDs must be unique")
        documents_by_id[document.document_id] = document
        if not isinstance(document.text, str):
            raise ValueError("complete retrieval artifacts require source text")
        _finite(document.score, name="document score")
        for name in ("url", "date", "language", "content_source", "english_text"):
            value = getattr(document, name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{name} must be a string or None")
        if document.content_local_cache_hit is not None and not isinstance(
            document.content_local_cache_hit, bool
        ):
            raise TypeError("content_local_cache_hit must be a boolean or None")
        if document.language is not None and document.language not in _SUPPORTED_LANGUAGES:
            raise ValueError("document language is not a supported corpus split")
        if document.content_source in _CORPUS_CONTENT_SOURCES:
            if document.language is None:
                raise ValueError("corpus content requires a supported language")
        elif document.content_source == "ragtime_api":
            if (
                document.language is not None
                or document.url is not None
                or document.date is not None
                or document.english_text is not None
            ):
                raise ValueError("ragtime_api content has invalid corpus metadata")
            if not isinstance(document.content_local_cache_hit, bool):
                raise ValueError("ragtime_api content must record cache provenance")
        else:
            raise ValueError("document content source is not supported")

    failures = tuple(result.hydration_failures)
    if any(not isinstance(failure, HydrationFailure) for failure in failures):
        raise TypeError("hydration_failures must be HydrationFailure values")
    if failures:
        raise ValueError("operational hydration failures are not cacheable")
    missing = tuple(result.missing_translation_ids)
    if len(set(missing)) != len(missing):
        raise ValueError("missing translation IDs must be unique")
    for document_id in missing:
        if document_id not in documents_by_id:
            raise ValueError("missing translation ID is not a hydrated document")
        document = documents_by_id[document_id]
        if document.english_text is not None or document.language == "eng":
            raise ValueError("translation status is inconsistent with document metadata")
    for document in documents:
        if (
            document.language != "eng"
            and document.english_text is None
            and document.document_id not in missing
        ):
            raise ValueError("missing English translation must be explicit")


def _to_wire(result: RetrievalResult) -> RetrievalResultWire:
    search = result.search_result
    return RetrievalResultWire(
        search_result=RetrievalPipelineWire(
            query=search.query,
            pipeline=search.pipeline,
            hits=tuple(
                RetrievalHitWire(rank=hit.rank, document_id=hit.document_id, score=float(hit.score))
                for hit in search.hits
            ),
            processed=search.processed,
            local_cache_hit=search.local_cache_hit,
            remote_cached=search.remote_cached,
            server_timestamp=None if search.server_timestamp is None else float(search.server_timestamp),
            cache_stored_at=None if search.cache_stored_at is None else float(search.cache_stored_at),
            extra=_canonical_json(search.extra),
        ),
        documents=tuple(
            RetrievalDocumentWire(
                rank=document.rank,
                document_id=document.document_id,
                score=float(document.score),
                text=document.text,
                url=document.url,
                date=document.date,
                language=document.language,
                content_source=document.content_source,
                content_local_cache_hit=document.content_local_cache_hit,
                english_text=document.english_text,
            )
            for document in result.documents
        ),
        hydration_failures=tuple(
            RetrievalFailureWire(split=failure.split, message=failure.message)
            for failure in result.hydration_failures
        ),
        missing_translation_ids=tuple(result.missing_translation_ids),
    )


def _from_wire(wire: RetrievalResultWire) -> RetrievalResult:
    search = wire.search_result
    result = RetrievalResult(
        search_result=PipelineResult(
            query=search.query,
            pipeline=search.pipeline,
            hits=tuple(
                SearchHit(rank=hit.rank, document_id=hit.document_id, score=hit.score)
                for hit in search.hits
            ),
            processed=search.processed,
            local_cache_hit=search.local_cache_hit,
            remote_cached=search.remote_cached,
            server_timestamp=search.server_timestamp,
            cache_stored_at=search.cache_stored_at,
            extra=dict(search.extra),
        ),
        documents=tuple(
            RetrievedDocument(
                rank=document.rank,
                document_id=document.document_id,
                score=document.score,
                text=document.text,
                url=document.url,
                date=document.date,
                language=document.language,
                content_source=document.content_source,
                content_local_cache_hit=document.content_local_cache_hit,
                english_text=document.english_text,
            )
            for document in wire.documents
        ),
        hydration_failures=tuple(
            HydrationFailure(split=failure.split, message=failure.message)
            for failure in wire.hydration_failures
        ),
        missing_translation_ids=tuple(wire.missing_translation_ids),
    )
    return result


def _canonical_envelope_bytes(envelope: RetrievalEnvelope) -> bytes:
    """Encode a decoded envelope in the one canonical wire representation."""

    search = envelope.result.search_result
    canonical_result = RetrievalResultWire(
        search_result=RetrievalPipelineWire(
            query=search.query,
            pipeline=search.pipeline,
            hits=search.hits,
            processed=search.processed,
            local_cache_hit=search.local_cache_hit,
            remote_cached=search.remote_cached,
            server_timestamp=search.server_timestamp,
            cache_stored_at=search.cache_stored_at,
            extra=_canonical_json(search.extra),
        ),
        documents=envelope.result.documents,
        hydration_failures=envelope.result.hydration_failures,
        missing_translation_ids=envelope.result.missing_translation_ids,
    )
    return msgspec.msgpack.encode(
        RetrievalEnvelope(
            schema_version=envelope.schema_version,
            retriever_fingerprint=envelope.retriever_fingerprint,
            request_digest=envelope.request_digest,
            stored_at=envelope.stored_at,
            result=canonical_result,
        )
    )


@dataclass(frozen=True, slots=True)
class _DecodedArtifact:
    result: RetrievalResult
    stored_at: float


class RetrievalArtifactCache:
    """Typed cache adapter for one complete hydrated retrieval per identity."""

    def __init__(self, store: ArtifactStoreLike, *, clock: Callable[[], float] = time.time) -> None:
        if not callable(getattr(store, "read_validated", None)) or not callable(
            getattr(store, "write", None)
        ):
            raise TypeError("store must implement read_validated and write")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.store = store
        self._clock = clock

    @staticmethod
    def _decode_error(message: str) -> ArtifactValidationError:
        return ArtifactValidationError(message)

    def _decoder(self, request: RetrievalRequestIdentity) -> Callable[[bytes], _DecodedArtifact]:
        def decode(payload: bytes) -> _DecodedArtifact:
            try:
                envelope = msgspec.msgpack.decode(payload, type=RetrievalEnvelope)
            except (msgspec.DecodeError, msgspec.ValidationError, TypeError, ValueError):
                raise self._decode_error("cached retrieval envelope is invalid") from None
            try:
                _validate_schema_version(envelope.schema_version)
                if (
                    envelope.retriever_fingerprint != request.retriever_fingerprint
                    or envelope.request_digest != request.request_digest
                ):
                    raise ValueError("cached retrieval identity is invalid")
                if _canonical_envelope_bytes(envelope) != payload:
                    raise ValueError("cached retrieval wire encoding is not canonical")
                _finite(envelope.stored_at, name="stored_at")
                result = _from_wire(envelope.result)
                _validate_result(request, result)
            except Exception:
                raise self._decode_error("cached retrieval payload is invalid") from None
            return _DecodedArtifact(result=result, stored_at=float(envelope.stored_at))

        return decode

    def get(self, request: RetrievalRequestIdentity) -> RetrievalResult | None:
        if not isinstance(request, RetrievalRequestIdentity):
            raise TypeError("request must be a RetrievalRequestIdentity")
        decoded = self.store.read_validated(request.object_key, self._decoder(request))
        if decoded is None:
            return None
        if not isinstance(decoded, _DecodedArtifact):
            raise ArtifactValidationError("cached retrieval decoder returned an invalid value")
        reference = RetrievalArtifactReference(
            object_key=request.object_key,
            request_digest=request.request_digest,
            retriever_fingerprint=request.retriever_fingerprint,
            stored_at=decoded.stored_at,
            cache_hit=True,
        )
        return RetrievalResult(
            search_result=decoded.result.search_result,
            documents=decoded.result.documents,
            hydration_failures=decoded.result.hydration_failures,
            missing_translation_ids=decoded.result.missing_translation_ids,
            retrieval_artifact=reference,
        )

    def put(self, request: RetrievalRequestIdentity, result: RetrievalResult) -> RetrievalResult:
        if not isinstance(request, RetrievalRequestIdentity):
            raise TypeError("request must be a RetrievalRequestIdentity")
        _validate_result(request, result)
        stored_at = _finite(self._clock(), name="cache clock")
        envelope = RetrievalEnvelope(
            schema_version=RETRIEVAL_SCHEMA_VERSION,
            retriever_fingerprint=request.retriever_fingerprint,
            request_digest=request.request_digest,
            stored_at=stored_at,
            result=_to_wire(result),
        )
        stored = self.store.write(request.object_key, msgspec.msgpack.encode(envelope))
        if stored is not True:
            return result
        reference = RetrievalArtifactReference(
            object_key=request.object_key,
            request_digest=request.request_digest,
            retriever_fingerprint=request.retriever_fingerprint,
            stored_at=stored_at,
            cache_hit=False,
        )
        return RetrievalResult(
            search_result=result.search_result,
            documents=result.documents,
            hydration_failures=result.hydration_failures,
            missing_translation_ids=result.missing_translation_ids,
            retrieval_artifact=reference,
        )


__all__ = [
    "CLIENT_BEHAVIOR_VERSION",
    "ENDPOINT_DEPTH_BEHAVIOR_VERSION",
    "FALLBACK_BEHAVIOR_VERSION",
    "HYDRATION_BEHAVIOR_VERSION",
    "RETRIEVAL_OBJECT_KEY_PREFIX",
    "RETRIEVAL_SCHEMA_VERSION",
    "RETRIEVER_BEHAVIOR_VERSION",
    "TRANSLATION_BEHAVIOR_VERSION",
    "RetrievalArtifactCache",
    "RetrievalArtifactReference",
    "RetrievalDocumentWire",
    "RetrievalEnvelope",
    "RetrievalFailureWire",
    "RetrievalHitWire",
    "RetrievalPipelineWire",
    "RetrievalRequestIdentity",
    "RetrievalResultWire",
    "retrieval_object_key",
    "retriever_fingerprint",
]
