"""Typed MessagePack response artifacts backed by :mod:`artifact_store`.

This module owns only response identity and wire validation.  The artifact
store owns local/remote lookup, repair, and transport failures; callers pass
the same endpoint validator used for a live response into :meth:`get`.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar
from urllib.parse import urlsplit

import msgspec

from .artifact_store import (
    ArtifactStoreLike,
    ArtifactStoreRemoteError,
    ArtifactValidationError,
)

CACHE_SCHEMA_VERSION = 1
DEFAULT_CACHE_NAMESPACE = "ragtime2-2026-v1"
_OBJECT_KEY_PREFIX = "api-responses/v1"

T = TypeVar("T")


class ResponseEnvelope(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Strict wire envelope for one endpoint response artifact."""

    schema_version: int
    namespace: str
    operation: str
    base_url_digest: bytes
    request_digest: bytes
    stored_at: float
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CacheEntry(Generic[T]):
    """A validated cached response and the time it was stored."""

    value: T
    stored_at: float

# These names are never part of a result-bearing request.  Rejecting them at
# canonicalization is a defense-in-depth boundary: an accidentally forwarded
# Authorization/token field cannot silently create an identity collision or
# be persisted in an artifact.
_CREDENTIAL_FIELDS = frozenset(
    {
        "authorization",
        "authentication",
        "auth",
        "auth_header",
        "credential",
        "credentials",
        "password",
        "passwd",
        "secret",
        "token",
        "api_key",
        "apikey",
    }
)


def _normalized_field_name(name: str) -> str:
    # Split camelCase before replacing separators so accessToken, access-token,
    # and access_token all have one security classification.
    split_camel = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", split_camel).strip("_")
    return re.sub(r"_+", "_", normalized).casefold()


def _is_credential_field(name: str) -> bool:
    normalized = _normalized_field_name(name)
    # ``token_count`` is a semantic measurement, not a bearer credential.  It
    # is the one common token-prefixed field that must remain representable;
    # exact token fields and provider/header spellings below stay rejected.
    if normalized == "token_count":
        return False
    if normalized in _CREDENTIAL_FIELDS:
        return True
    parts = normalized.split("_")
    # A token/secret/password/auth/credential component is sensitive even when
    # prefixed by a provider-specific name (client_secret, refresh_token, ...).
    if any(
        part in {
            "auth",
            "authorization",
            "authentication",
            "credential",
            "credentials",
            "password",
            "passwd",
            "secret",
            "token",
            "apikey",
        }
        for part in parts
    ):
        return True
    # API keys use a two-part marker; accepting a leading x-/client-/server-
    # component covers the common header and configuration spellings.
    return any(
        parts[index : index + 2] == ["api", "key"]
        for index in range(len(parts) - 1)
    )


_PUBLIC_METADATA_SENSITIVE_PARTS = frozenset(
    {
        "auth",
        "authorization",
        "authentication",
        "body",
        "credential",
        "credentials",
        "password",
        "passwd",
        "query",
        "response",
        "secret",
        "url",
    }
)
_PUBLIC_METADATA_BUCKET_PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_PUBLIC_METADATA_PREFIX_PART = re.compile(
    r"[A-Za-z0-9_~%+-][A-Za-z0-9._~%+-]*\Z"
)


def _safe_public_metadata_value(field: str, value: Any) -> str:
    """Validate one injected-store metadata value without echoing it."""

    try:
        if field == "local_dir" and isinstance(value, os.PathLike):
            value = os.fspath(value)
        if not isinstance(value, str):
            raise ValueError
        if field == "prefix" and value == "":
            return value
        if not value or any(
            character.isspace() or ord(character) < 32 for character in value
        ):
            raise ValueError
        if "//" in value and "://" in value:
            raise ValueError
        if any(character in value for character in "?#\\=@{}[]<>\"';&:"):
            raise ValueError
        normalized = _normalized_field_name(value)
        if _is_credential_field(value) or any(
            part in _PUBLIC_METADATA_SENSITIVE_PARTS
            for part in normalized.split("_")
        ):
            raise ValueError
        if field == "bucket_id":
            parts = value.split("/")
            if len(parts) != 2 or any(
                not _PUBLIC_METADATA_BUCKET_PART.fullmatch(part) for part in parts
            ):
                raise ValueError
        elif field == "prefix":
            parts = value.split("/")
            if any(
                not part or part in {".", ".."}
                or not _PUBLIC_METADATA_PREFIX_PART.fullmatch(part)
                for part in parts
            ):
                raise ValueError
        elif field == "local_dir":
            if not os.path.isabs(value):
                raise ValueError
        else:
            raise ValueError
    except Exception:
        raise ValueError("artifact cache metadata is invalid") from None
    return value


def _normalize_base_url(base_url: str) -> str:
    if not isinstance(base_url, str):
        raise TypeError("base URL must be a string")
    normalized = base_url.strip().rstrip("/")
    if not normalized:
        raise ValueError("base URL must not be empty")
    if "?" in normalized or "#" in normalized:
        raise ValueError("base URL must not contain credentials, a query, or a fragment")
    scheme_separator = normalized.find("://")
    if scheme_separator >= 0:
        authority_end = len(normalized)
        for delimiter in "/?#":
            delimiter_index = normalized.find(delimiter, scheme_separator + 3)
            if delimiter_index >= 0:
                authority_end = min(authority_end, delimiter_index)
        authority = normalized[scheme_separator + 3 : authority_end]
        if any(character.isspace() or ord(character) < 32 for character in authority):
            raise ValueError("base URL authority is invalid")
    try:
        parsed = urlsplit(normalized)
        # Force validation of malformed numeric ports at this shared identity
        # boundary rather than deferring it to an HTTP client.
        parsed.port
    except ValueError:
        raise ValueError("base URL must be an absolute HTTP(S) URL") from None
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.hostname is None
        or parsed.netloc.rsplit("@", 1)[-1].endswith(":")
    ):
        raise ValueError("base URL must be an absolute HTTP(S) URL")
    if (
        parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError("base URL must not contain credentials, a query, or a fragment")
    return normalized


def _validate_namespace(namespace: str) -> str:
    if not isinstance(namespace, str):
        raise TypeError("cache namespace must be a string")
    value = namespace.strip()
    if not value:
        raise ValueError("cache namespace must not be empty")
    return value


def _validate_operation(operation: str) -> str:
    if not isinstance(operation, str) or not operation or operation != operation.strip():
        raise ValueError("cache operation must be one relative path segment")
    if operation in {".", ".."} or any(
        character in operation for character in "/\\?#"
    ):
        raise ValueError("cache operation must be one relative path segment")
    if any(ord(character) < 32 for character in operation):
        raise ValueError("cache operation contains a control character")
    return operation


def canonicalize_json(value: Any) -> Any:
    """Return recursively sorted JSON-like data suitable for MessagePack.

    Mapping keys must be strings, mapping order is discarded recursively, list
    order is retained, and credential-bearing fields are rejected before an
    identity or persisted artifact is produced.
    Unsupported values and non-finite floats are rejected before encoding.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("cache values must contain only finite numbers")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError("cache mapping keys must be strings")
            if _is_credential_field(key):
                raise ValueError("cache mappings must not contain credential fields")
            normalized[key] = canonicalize_json(nested)
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, list):
        return [canonicalize_json(item) for item in value]
    raise TypeError(
        "cache values must be JSON-like (null, boolean, number, string, list, or object)"
    )


def canonical_request_bytes(
    base_url: str,
    operation: str,
    request: Mapping[str, Any],
    namespace: str = DEFAULT_CACHE_NAMESPACE,
) -> bytes:
    """Encode the canonical request identity as deterministic MessagePack."""

    normalized_namespace = _validate_namespace(namespace)
    normalized_base_url = _normalize_base_url(base_url)
    normalized_operation = _validate_operation(operation)
    canonical_request = canonicalize_json(request)
    if not isinstance(canonical_request, dict):  # Mapping is checked defensively.
        raise TypeError("cache request must be an object")
    identity = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "namespace": normalized_namespace,
        "base_url": normalized_base_url,
        "operation": normalized_operation,
        "request": canonical_request,
    }
    return msgspec.msgpack.encode(identity)


def request_digest(
    base_url: str,
    operation: str,
    request: Mapping[str, Any],
    namespace: str = DEFAULT_CACHE_NAMESPACE,
) -> bytes:
    """Return the binary SHA-256 digest of a canonical request identity."""

    return hashlib.sha256(
        canonical_request_bytes(base_url, operation, request, namespace)
    ).digest()


def base_url_digest(base_url: str) -> bytes:
    """Return the binary SHA-256 digest of a normalized base URL."""

    return hashlib.sha256(_normalize_base_url(base_url).encode("utf-8")).digest()


def response_object_key(operation: str, digest: bytes | bytearray | str) -> str:
    """Build the immutable artifact path for an operation and request hash."""

    normalized_operation = _validate_operation(operation)
    if isinstance(digest, str):
        try:
            digest_bytes = bytes.fromhex(digest)
        except ValueError:
            raise ValueError("request digest must be 32 SHA-256 bytes") from None
    elif isinstance(digest, (bytes, bytearray)):
        digest_bytes = bytes(digest)
    else:
        raise TypeError("request digest must be bytes or hexadecimal text")
    if len(digest_bytes) != hashlib.sha256().digest_size:
        raise ValueError("request digest must be 32 SHA-256 bytes")
    digest_hex = digest_bytes.hex()
    return f"{_OBJECT_KEY_PREFIX}/{normalized_operation}/{digest_hex[:2]}/{digest_hex}.mpk"


def _stored_at(value: float | int, *, source: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{source} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{source} must be a finite number")
    return number


class ResponseCache:
    """Typed response cache over an :class:`ArtifactStoreLike` instance."""

    def __init__(
        self,
        store: ArtifactStoreLike,
        *,
        namespace: str = DEFAULT_CACHE_NAMESPACE,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.store = store
        self.namespace = _validate_namespace(namespace)
        self._clock = clock

    def object_key(
        self,
        *,
        base_url: str,
        operation: str,
        request: Mapping[str, Any],
    ) -> str:
        """Return the exact private-artifact key for a request."""

        return response_object_key(
            operation,
            request_digest(
                base_url,
                operation,
                request,
                self.namespace,
            ),
        )

    def _identity(
        self,
        *,
        base_url: str,
        operation: str,
        request: Mapping[str, Any],
    ) -> tuple[bytes, bytes]:
        normalized_base_url = _normalize_base_url(base_url)
        normalized_operation = _validate_operation(operation)
        digest = request_digest(
            normalized_base_url,
            normalized_operation,
            request,
            self.namespace,
        )
        return base_url_digest(normalized_base_url), digest

    @staticmethod
    def _decode_error(message: str) -> ArtifactValidationError:
        return ArtifactValidationError(message)

    def _decoder(
        self,
        *,
        expected_base_url_digest: bytes,
        expected_request_digest: bytes,
        operation: str,
        validator: Callable[[Mapping[str, Any]], T],
    ) -> Callable[[bytes], CacheEntry[T]]:
        def decode(payload: bytes) -> CacheEntry[T]:
            try:
                envelope = msgspec.msgpack.decode(payload, type=ResponseEnvelope)
            except (msgspec.DecodeError, msgspec.ValidationError, TypeError, ValueError):
                raise self._decode_error("cached response envelope is invalid") from None
            if (
                envelope.schema_version != CACHE_SCHEMA_VERSION
                or envelope.namespace != self.namespace
                or envelope.operation != operation
                or envelope.base_url_digest != expected_base_url_digest
                or envelope.request_digest != expected_request_digest
                or len(envelope.base_url_digest) != hashlib.sha256().digest_size
                or len(envelope.request_digest) != hashlib.sha256().digest_size
            ):
                raise self._decode_error("cached response identity is invalid")
            try:
                stored_at = _stored_at(envelope.stored_at, source="stored_at")
                canonical_payload = canonicalize_json(envelope.payload)
                if not isinstance(canonical_payload, dict):
                    raise TypeError("cached payload must be an object")
                canonical_envelope = ResponseEnvelope(
                    schema_version=envelope.schema_version,
                    namespace=envelope.namespace,
                    operation=envelope.operation,
                    base_url_digest=envelope.base_url_digest,
                    request_digest=envelope.request_digest,
                    stored_at=stored_at,
                    payload=canonical_payload,
                )
                if msgspec.msgpack.encode(canonical_envelope) != payload:
                    raise ValueError("cached response wire encoding is not canonical")
                value = validator(canonical_payload)
            except Exception:
                # Domain validators deliberately have no cache-specific
                # dependency.  Convert every rejection into the one error
                # ArtifactStore uses to repair an invalid L1 candidate.
                raise self._decode_error("cached response payload is invalid") from None
            return CacheEntry(value=value, stored_at=stored_at)

        return decode

    def get(
        self,
        *,
        base_url: str,
        operation: str,
        request: Mapping[str, Any],
        ttl_seconds: float | None,
        validator: Callable[[Mapping[str, Any]], T],
    ) -> CacheEntry[T] | None:
        """Read, strictly validate, and then apply TTL to one response."""

        if ttl_seconds is not None:
            if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, (int, float)):
                raise ValueError("cache TTL must be finite and non-negative or None")
            if not math.isfinite(float(ttl_seconds)) or ttl_seconds < 0:
                raise ValueError("cache TTL must be finite and non-negative or None")
        if not callable(validator):
            raise TypeError("cache validator must be callable")

        expected_base_url_digest, expected_request_digest = self._identity(
            base_url=base_url,
            operation=operation,
            request=request,
        )
        normalized_operation = _validate_operation(operation)
        object_key = response_object_key(normalized_operation, expected_request_digest)
        try:
            entry = self.store.read_validated(
                object_key,
                self._decoder(
                    expected_base_url_digest=expected_base_url_digest,
                    expected_request_digest=expected_request_digest,
                    operation=normalized_operation,
                    validator=validator,
                ),
            )
        except (OSError, ArtifactStoreRemoteError):
            return None
        if entry is None:
            return None
        if ttl_seconds is not None:
            now = _stored_at(self._clock(), source="cache clock")
            if now - entry.stored_at >= float(ttl_seconds):
                return None
        return entry

    def put(
        self,
        *,
        base_url: str,
        operation: str,
        request: Mapping[str, Any],
        response: Mapping[str, Any],
        stored_at: float | None = None,
    ) -> float | None:
        """Encode and best-effort write one response without rereading it."""

        expected_base_url_digest, expected_request_digest = self._identity(
            base_url=base_url,
            operation=operation,
            request=request,
        )
        normalized_operation = _validate_operation(operation)
        canonical_response = canonicalize_json(response)
        if not isinstance(canonical_response, dict):
            raise TypeError("cache response must be an object")
        timestamp = (
            _stored_at(self._clock(), source="cache clock")
            if stored_at is None
            else _stored_at(stored_at, source="stored_at")
        )
        envelope = ResponseEnvelope(
            schema_version=CACHE_SCHEMA_VERSION,
            namespace=self.namespace,
            operation=normalized_operation,
            base_url_digest=expected_base_url_digest,
            request_digest=expected_request_digest,
            stored_at=timestamp,
            payload=canonical_response,
        )
        encoded = msgspec.msgpack.encode(envelope)
        try:
            stored = self.store.write(
                response_object_key(normalized_operation, expected_request_digest),
                encoded,
            )
        except (OSError, ArtifactStoreRemoteError):
            return None
        return timestamp if stored is True else None

    def evict_local(
        self,
        *,
        base_url: str,
        operation: str,
        request: Mapping[str, Any],
    ) -> None:
        """Evict exactly one L1 candidate without probing or changing L2."""

        _, expected_request_digest = self._identity(
            base_url=base_url,
            operation=operation,
            request=request,
        )
        normalized_operation = _validate_operation(operation)
        evict = getattr(self.store, "evict_local", None)
        if not callable(evict):
            raise ArtifactStoreRemoteError(
                "artifact store cannot evict a local cache candidate"
            )
        evict(response_object_key(normalized_operation, expected_request_digest))


def sanitized_cache_configuration(
    namespace: str, store: object | None
) -> dict[str, object]:
    """Return safe public cache metadata, including injected stores.

    Runtime clients expose this data in manifests and status surfaces.  An
    injected ``ArtifactStoreLike`` is not trusted merely because it satisfies
    the protocol, so all copied values are constrained to their expected
    path-like forms and failures use a value-free message.
    """

    try:
        if not isinstance(namespace, str) or not namespace.strip():
            raise ValueError
        normalized_namespace = _safe_public_metadata_value("prefix", namespace.strip())
        configuration: dict[str, object] = {"namespace": normalized_namespace}
        for field in ("bucket_id", "prefix", "local_dir"):
            try:
                value = getattr(store, field, None)
            except Exception:
                raise ValueError from None
            if value is not None:
                configuration[field] = _safe_public_metadata_value(field, value)
        return configuration
    except Exception:
        raise ValueError("artifact cache metadata is invalid") from None


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "DEFAULT_CACHE_NAMESPACE",
    "CacheEntry",
    "ResponseCache",
    "ResponseEnvelope",
    "base_url_digest",
    "canonical_request_bytes",
    "canonicalize_json",
    "request_digest",
    "response_object_key",
    "sanitized_cache_configuration",
]
