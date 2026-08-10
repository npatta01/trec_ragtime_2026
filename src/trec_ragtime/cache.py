"""Typed MessagePack response artifacts backed by :mod:`artifact_store`.

This module owns only response identity and wire validation.  The artifact
store owns local/remote lookup, repair, and transport failures; callers pass
the same endpoint validator used for a live response into :meth:`get`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import stat
import time
from collections.abc import Callable, Mapping
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import urlsplit

import msgspec

from .artifact_store import ArtifactStoreLike, ArtifactValidationError

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
class CacheEntry:
    """A decoded cached response and the time it was stored."""

    payload: dict[str, Any]
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


def _normalize_base_url(base_url: str) -> str:
    if not isinstance(base_url, str):
        raise TypeError("base URL must be a string")
    normalized = base_url.strip().rstrip("/")
    if not normalized:
        raise ValueError("base URL must not be empty")
    parsed = urlsplit(normalized)
    if (
        parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
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


# Explicit aliases make the identity primitives convenient to use from
# importers without exposing implementation details of ResponseCache.
canonical_request = canonical_request_bytes
request_identity_digest = request_digest
object_key_for_request = response_object_key


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
    ) -> Callable[[bytes], CacheEntry]:
        def decode(payload: bytes) -> CacheEntry:
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
                validator(canonical_payload)
            except Exception:
                # Domain validators deliberately have no cache-specific
                # dependency.  Convert every rejection into the one error
                # ArtifactStore uses to repair an invalid L1 candidate.
                raise self._decode_error("cached response payload is invalid") from None
            return CacheEntry(payload=canonical_payload, stored_at=stored_at)

        return decode

    def get(
        self,
        *,
        base_url: str,
        operation: str,
        request: Mapping[str, Any],
        ttl_seconds: float | None,
        validator: Callable[[Mapping[str, Any]], T],
    ) -> CacheEntry | None:
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
        entry = self.store.read_validated(
            object_key,
            self._decoder(
                expected_base_url_digest=expected_base_url_digest,
                expected_request_digest=expected_request_digest,
                operation=normalized_operation,
                validator=validator,
            ),
        )
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
    ) -> float:
        """Encode and write one response exactly once, without rereading it."""

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
        self.store.write(response_object_key(normalized_operation, expected_request_digest), encoded)
        return timestamp


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "DEFAULT_CACHE_NAMESPACE",
    "CacheEntry",
    "ResponseCache",
    "ResponseEnvelope",
    "base_url_digest",
    "canonical_request",
    "canonical_request_bytes",
    "canonicalize_json",
    "object_key_for_request",
    "request_digest",
    "request_identity_digest",
    "response_object_key",
]


# Temporary compatibility for the not-yet-migrated HTTP clients.  Task 3
# removes these runtime SQLite surfaces after those clients consume
# ResponseCache; the typed artifact path above is the new implementation.
DEFAULT_CACHE_FILENAME = "api-v1.sqlite3"


def _absolute_path(value: str | os.PathLike[str], setting: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{setting} must be an absolute path")
    return path


def default_cache_path() -> Path:
    from .paths import default_project_cache_dir

    explicit_path = os.getenv("RAGTIME_CACHE_PATH", "").strip()
    if explicit_path:
        return _absolute_path(explicit_path, "RAGTIME_CACHE_PATH")
    explicit_dir = os.getenv("RAGTIME_CACHE_DIR", "").strip()
    if explicit_dir:
        return _absolute_path(explicit_dir, "RAGTIME_CACHE_DIR") / DEFAULT_CACHE_FILENAME
    return default_project_cache_dir() / DEFAULT_CACHE_FILENAME


class SQLiteResponseCache:
    """Legacy cache retained only until the runtime-client migration."""

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        namespace: str = DEFAULT_CACHE_NAMESPACE,
        busy_timeout_seconds: float = 30.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = (
            _absolute_path(path, "cache path")
            if path is not None
            else default_cache_path()
        )
        self.namespace = namespace.strip()
        if not self.namespace:
            raise ValueError("cache namespace must not be empty")
        if busy_timeout_seconds <= 0:
            raise ValueError("busy timeout must be positive")
        self._busy_timeout_seconds = busy_timeout_seconds
        self._clock = clock
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=self._busy_timeout_seconds)
        connection.execute(
            f"PRAGMA busy_timeout = {int(self._busy_timeout_seconds * 1000)}"
        )
        return connection

    def _initialize(self) -> None:
        from .paths import ensure_private_cache_dir

        ensure_private_cache_dir(self.path.parent)
        open_flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            open_flags |= os.O_NOFOLLOW
        file_descriptor = os.open(self.path, open_flags, 0o600)
        try:
            file_status = os.fstat(file_descriptor)
            if not stat.S_ISREG(file_status.st_mode):
                raise PermissionError(f"cache path is not a regular file: {self.path}")
            os.fchmod(file_descriptor, 0o600)
        finally:
            os.close(file_descriptor)
        with closing(self._connect()) as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS responses (
                    cache_key TEXT PRIMARY KEY,
                    schema_version INTEGER NOT NULL,
                    namespace TEXT NOT NULL,
                    base_url TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    stored_at REAL NOT NULL
                )
                """
            )
            connection.commit()

    @staticmethod
    def _canonical_json(value: Mapping[str, Any]) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    def _identity(
        self,
        *,
        base_url: str,
        operation: str,
        request: Mapping[str, Any],
    ) -> tuple[str, str]:
        request_json = self._canonical_json(request)
        identity = self._canonical_json(
            {
                "base_url": base_url,
                "namespace": self.namespace,
                "operation": operation,
                "request": request,
                "schema_version": CACHE_SCHEMA_VERSION,
            }
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest(), request_json

    def get(
        self,
        *,
        base_url: str,
        operation: str,
        request: Mapping[str, Any],
        ttl_seconds: float | None,
    ) -> CacheEntry | None:
        if ttl_seconds is not None and ttl_seconds < 0:
            raise ValueError("cache TTL must be non-negative or None")
        cache_key, _ = self._identity(
            base_url=base_url, operation=operation, request=request
        )
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT response_json, stored_at FROM responses WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            if row is None:
                return None
            response_json, stored_at = row
            if ttl_seconds is not None and self._clock() - stored_at >= ttl_seconds:
                return None
            try:
                payload = json.loads(response_json)
            except (TypeError, json.JSONDecodeError):
                connection.execute("DELETE FROM responses WHERE cache_key = ?", (cache_key,))
                connection.commit()
                return None
            if not isinstance(payload, dict):
                connection.execute("DELETE FROM responses WHERE cache_key = ?", (cache_key,))
                connection.commit()
                return None
            return CacheEntry(payload=payload, stored_at=float(stored_at))

    def put(
        self,
        *,
        base_url: str,
        operation: str,
        request: Mapping[str, Any],
        response: Mapping[str, Any],
    ) -> float:
        cache_key, request_json = self._identity(
            base_url=base_url, operation=operation, request=request
        )
        response_json = self._canonical_json(response)
        stored_at = self._clock()
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO responses (
                        cache_key, schema_version, namespace, base_url, operation,
                        request_json, response_json, stored_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(cache_key) DO UPDATE SET
                        response_json = excluded.response_json,
                        stored_at = excluded.stored_at
                    """,
                    (
                        cache_key,
                        CACHE_SCHEMA_VERSION,
                        self.namespace,
                        base_url,
                        operation,
                        request_json,
                        response_json,
                        stored_at,
                    ),
                )
        return stored_at

    def clear(self) -> None:
        with closing(self._connect()) as connection:
            with connection:
                connection.execute("DELETE FROM responses WHERE namespace = ?", (self.namespace,))

    def delete(
        self,
        *,
        base_url: str,
        operation: str,
        request: Mapping[str, Any],
    ) -> None:
        cache_key, _ = self._identity(
            base_url=base_url, operation=operation, request=request
        )
        with closing(self._connect()) as connection:
            with connection:
                connection.execute("DELETE FROM responses WHERE cache_key = ?", (cache_key,))
