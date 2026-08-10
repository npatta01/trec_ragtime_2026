"""One-time, read-only migration of legacy SQLite response artifacts.

SQLite is deliberately imported only in this explicit offline module. Runtime
clients use :class:`~trec_ragtime.cache.ResponseCache` over artifact storage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .artifact_store import ArtifactStore
from .cache import (
    DEFAULT_CACHE_NAMESPACE,
    ResponseCache,
    _is_credential_field,
    canonicalize_json,
)
from .client import RagtimeClient
from .huggingface import (
    DEFAULT_HF_CONFIG,
    DEFAULT_HF_DATASET,
    DEFAULT_HF_DOCUMENT_CACHE_TTL_SECONDS,
    DEFAULT_HF_SPLITS,
    HF_CACHE_OPERATION,
    HuggingFaceCorpus,
)

SUPPORTED_OPERATIONS = ("pipeline", "content", HF_CACHE_OPERATION)
_LEGACY_COLUMNS = (
    "cache_key",
    "schema_version",
    "namespace",
    "base_url",
    "operation",
    "request_json",
    "response_json",
    "stored_at",
)
_OPERATION_TTLS: dict[str, float | None] = {
    "pipeline": None,
    "content": None,
    HF_CACHE_OPERATION: DEFAULT_HF_DOCUMENT_CACHE_TTL_SECONDS,
}
_SAFE_BUCKET_PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_SAFE_PREFIX_PART = re.compile(r"[A-Za-z0-9_~%+-][A-Za-z0-9._~%+-]*\Z")
_SENSITIVE_METADATA_PARTS = frozenset(
    {
        "auth",
        "authorization",
        "authentication",
        "bearer",
        "body",
        "credential",
        "credentials",
        "key",
        "password",
        "passwd",
        "query",
        "request",
        "response",
        "secret",
        "text",
        "token",
        "url",
    }
)


class CacheImportError(RuntimeError):
    """Raised when an explicit cache migration cannot safely complete."""


@dataclass(frozen=True, slots=True)
class ImportSummary:
    """Sanitized migration counts and immutable source provenance."""

    source_sha256: str
    source_size: int
    imported: dict[str, int]
    skipped: dict[str, int]
    verified: dict[str, int]
    skipped_reasons: dict[str, int]

    @property
    def counts(self) -> dict[str, dict[str, int]]:
        """Return the per-operation counts used by the migration manifest."""

        operations = set(self.imported) | set(self.skipped) | set(self.verified)
        return {
            operation: {
                "imported": self.imported.get(operation, 0),
                "skipped": self.skipped.get(operation, 0),
                "verified": self.verified.get(operation, 0),
            }
            for operation in sorted(operations)
        }


@dataclass(frozen=True, slots=True)
class _ValidatedRow:
    base_url: str
    operation: str
    request: dict[str, Any]
    response: dict[str, Any]
    stored_at: float
    ttl_seconds: float | None
    validator: Callable[[Mapping[str, Any]], Any]


def _source_bytes(source: Path) -> tuple[bytes, str, int]:
    if not source.is_absolute():
        raise ValueError("SQLite source path must be absolute")
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)
    payload = source.read_bytes()
    return payload, hashlib.sha256(payload).hexdigest(), len(payload)


def _read_only_connection(source: Path) -> sqlite3.Connection:
    # Keep this exact URI form: mode=ro prevents an absent path from being
    # created and prevents any migration statement from mutating the source.
    return sqlite3.connect(f"file:{source}?mode=ro", uri=True)


def _validate_legacy_schema(connection: sqlite3.Connection) -> None:
    cursor: sqlite3.Cursor | None = None
    try:
        cursor = connection.execute("PRAGMA table_info(responses)")
        columns = tuple(row[1] for row in cursor.fetchall())
    except sqlite3.DatabaseError:
        raise CacheImportError("source SQLite schema could not be read") from None
    finally:
        if cursor is not None:
            cursor.close()
    if columns != _LEGACY_COLUMNS:
        raise CacheImportError("source SQLite schema is not the legacy responses table")


def _json_object(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, str):
        raise ValueError("JSON field is not text")
    decoded = json.loads(raw)
    if not isinstance(decoded, dict):
        raise ValueError("JSON field is not an object")
    return decoded


def _nonempty_string(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("field is not a non-empty string")
    return value


def _finite_timestamp(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("stored_at is not finite")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("stored_at is not finite")
    return number


def _pipeline_validator(
    request: Mapping[str, Any],
) -> Callable[[Mapping[str, Any]], Any]:
    query = _nonempty_string(request.get("query"))
    pipeline = _nonempty_string(request.get("pipeline"))

    def validate(payload: Mapping[str, Any]) -> Any:
        return RagtimeClient._parse_pipeline_payload(
            payload,
            query=query,
            pipeline=pipeline,
        )

    return validate


def _content_validator(
    request: Mapping[str, Any],
) -> Callable[[Mapping[str, Any]], Any]:
    collection = _nonempty_string(request.get("collection"))
    document_id = _nonempty_string(request.get("id"))

    def validate(payload: Mapping[str, Any]) -> Any:
        return RagtimeClient._parse_content_payload(
            payload,
            document_id=document_id,
            collection=collection,
        )

    return validate


def _huggingface_validator(
    request: Mapping[str, Any],
) -> Callable[[Mapping[str, Any]], Any]:
    expected_request = {
        "config": DEFAULT_HF_CONFIG,
        "dataset": DEFAULT_HF_DATASET,
        "id": _nonempty_string(request.get("id")),
        "splits": list(DEFAULT_HF_SPLITS),
        "variant": "original",
    }
    if dict(request) != expected_request:
        raise ValueError("Dataset Viewer request identity is not canonical")

    # The parser is pure domain validation. Constructing this object without
    # __init__ avoids a Requests session and therefore cannot contact the Hub.
    parser = HuggingFaceCorpus.__new__(HuggingFaceCorpus)
    parser.dataset = DEFAULT_HF_DATASET
    parser.config = DEFAULT_HF_CONFIG
    parser.splits = tuple(DEFAULT_HF_SPLITS)

    def validate(payload: Mapping[str, Any]) -> Any:
        return parser._parse_cached_payload(
            payload,
            expected_document_id=expected_request["id"],
        )

    return validate


def _validator_for(
    operation: str, request: Mapping[str, Any]
) -> Callable[[Mapping[str, Any]], Any]:
    if operation == "pipeline":
        return _pipeline_validator(request)
    if operation == "content":
        return _content_validator(request)
    if operation == HF_CACHE_OPERATION:
        return _huggingface_validator(request)
    raise ValueError("unsupported operation")


def _validate_row(
    row: tuple[Any, ...], *, canonical_namespace: str
) -> _ValidatedRow:
    if len(row) != len(_LEGACY_COLUMNS):
        raise ValueError("legacy row has the wrong shape")
    _, schema_version, namespace, base_url, operation, request_json, response_json, stored_at = row
    if isinstance(schema_version, bool) or schema_version != 1:
        raise ValueError("unsupported schema version")
    if namespace != canonical_namespace:
        raise ValueError("non-canonical namespace")
    if operation not in SUPPORTED_OPERATIONS:
        raise ValueError("unsupported operation")
    normalized_base_url = _nonempty_string(base_url).strip().rstrip("/")
    request = _json_object(request_json)
    response = _json_object(response_json)
    timestamp = _finite_timestamp(stored_at)
    validator = _validator_for(operation, request)
    # Run the same endpoint/domain parser used by live responses before any
    # destination write. ResponseCache will independently canonicalize it.
    validator(response)
    return _ValidatedRow(
        base_url=normalized_base_url,
        operation=operation,
        request=request,
        response=response,
        stored_at=timestamp,
        ttl_seconds=_OPERATION_TTLS[operation],
        validator=validator,
    )


def _safe_destination(cache: ResponseCache) -> dict[str, str]:
    def safe_value(field: str, value: Any) -> str:
        if not isinstance(value, str):
            raise CacheImportError("destination metadata is not safe to record")
        if field == "prefix" and value == "":
            return value
        if not value or any(
            character.isspace() or ord(character) < 32 for character in value
        ):
            raise CacheImportError("destination metadata is not safe to record")
        if any(character in value for character in "?#\\=@{}[]<>\"';&:"):
            raise CacheImportError("destination metadata is not safe to record")
        if "://" in value:
            raise CacheImportError("destination metadata is not safe to record")
        if _is_credential_field(value):
            raise CacheImportError("destination metadata is not safe to record")
        normalized = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").casefold()
        if any(part in _SENSITIVE_METADATA_PARTS for part in normalized.split("_")):
            raise CacheImportError("destination metadata is not safe to record")
        if field == "bucket_id":
            parts = value.split("/")
            if len(parts) != 2 or any(
                not _SAFE_BUCKET_PART.fullmatch(part) for part in parts
            ):
                raise CacheImportError("destination metadata is not safe to record")
        else:
            parts = value.split("/")
            if any(
                not part or part in {".", ".."} or not _SAFE_PREFIX_PART.fullmatch(part)
                for part in parts
            ):
                raise CacheImportError("destination metadata is not safe to record")
        return value

    destination: dict[str, str] = {"namespace": cache.namespace}
    store = getattr(cache, "store", None)
    for field in ("bucket_id", "prefix"):
        value = getattr(store, field, None)
        if value is None:
            continue
        destination[field] = safe_value(field, value)
    return destination


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path = path.expanduser()
    if not path.is_absolute():
        raise ValueError("manifest path must be absolute")
    if path.is_symlink():
        raise CacheImportError("manifest path must not be a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ).encode("utf-8") + b"\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            os.fchmod(handle.fileno(), 0o600)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def import_sqlite_cache(
    source: Path,
    cache: ResponseCache,
    manifest_path: Path,
    *,
    canonical_namespace: str = DEFAULT_CACHE_NAMESPACE,
) -> ImportSummary:
    """Import supported legacy rows from a source opened strictly read-only.

    Rows that are malformed, unsupported, expired, or rejected by the live
    endpoint validators are skipped with sanitized reason counts. Writes are
    followed by a separate validated destination read before a row is counted
    as verified. The source is never deleted or modified.
    """

    if not isinstance(cache, ResponseCache):
        raise TypeError("cache must be a ResponseCache")
    if not isinstance(canonical_namespace, str) or not canonical_namespace.strip():
        raise ValueError("canonical namespace must not be empty")
    if canonical_namespace != DEFAULT_CACHE_NAMESPACE:
        raise ValueError("canonical namespace must be the project default")
    if cache.namespace != canonical_namespace:
        raise ValueError("cache namespace does not match canonical namespace")
    source = Path(source).expanduser()
    destination = _safe_destination(cache)
    before, source_sha256, source_size = _source_bytes(source)
    imported = Counter({operation: 0 for operation in SUPPORTED_OPERATIONS})
    skipped = Counter({operation: 0 for operation in SUPPORTED_OPERATIONS})
    verified = Counter({operation: 0 for operation in SUPPORTED_OPERATIONS})
    skipped_reasons: Counter[str] = Counter()
    now = time.time()

    connection: sqlite3.Connection | None = None
    try:
        connection = _read_only_connection(source)
        with connection:
            _validate_legacy_schema(connection)
            rows = connection.execute(
                """
                SELECT cache_key, schema_version, namespace, base_url, operation,
                       request_json, response_json, stored_at
                FROM responses
                """
            )
            try:
                for raw_row in rows:
                    operation_value = raw_row[4]
                    operation = (
                        operation_value
                        if operation_value in SUPPORTED_OPERATIONS
                        else "unsupported"
                    )
                    try:
                        validated = _validate_row(
                            tuple(raw_row), canonical_namespace=canonical_namespace
                        )
                        if (
                            validated.ttl_seconds is not None
                            and now - validated.stored_at >= validated.ttl_seconds
                        ):
                            raise ValueError("expired row")
                    except CacheImportError:
                        raise
                    except Exception:
                        skipped[operation] += 1
                        skipped_reasons["row_rejected"] += 1
                        continue
                    try:
                        # The identity helper is a side-effect-free production
                        # preflight. It catches credential-bearing requests before
                        # the write while retaining a single canonical key path.
                        cache.object_key(
                            base_url=validated.base_url,
                            operation=validated.operation,
                            request=validated.request,
                        )
                        canonicalize_json(validated.response)
                    except Exception:
                        skipped[operation] += 1
                        skipped_reasons["row_rejected"] += 1
                        continue
                    try:
                        cache.put(
                            base_url=validated.base_url,
                            operation=validated.operation,
                            request=validated.request,
                            response=validated.response,
                            stored_at=validated.stored_at,
                        )
                        reread = cache.get(
                            base_url=validated.base_url,
                            operation=validated.operation,
                            request=validated.request,
                            ttl_seconds=validated.ttl_seconds,
                            validator=validated.validator,
                        )
                        if reread is None or reread.stored_at != validated.stored_at:
                            raise CacheImportError(
                                "destination artifact failed independent verification"
                            )
                    except CacheImportError:
                        raise
                    except Exception as exc:
                        raise CacheImportError(
                            "destination artifact failed independent verification"
                        ) from exc
                    imported[validated.operation] += 1
                    verified[validated.operation] += 1
            finally:
                rows.close()
    except sqlite3.OperationalError as exc:
        raise CacheImportError("source SQLite database could not be read") from exc
    except sqlite3.DatabaseError as exc:
        raise CacheImportError("source SQLite database is invalid") from exc
    finally:
        if connection is not None:
            connection.close()

    after = source.read_bytes()
    if after != before:
        raise CacheImportError("source SQLite bytes changed during import")

    summary = ImportSummary(
        source_sha256=source_sha256,
        source_size=source_size,
        imported=dict(imported),
        skipped=dict(skipped),
        verified=dict(verified),
        skipped_reasons=dict(skipped_reasons),
    )
    manifest = {
        "schema_version": 1,
        "source": {"sha256": summary.source_sha256, "size": summary.source_size},
        "destination": destination,
        "counts": summary.counts,
        "skipped_reasons": summary.skipped_reasons,
    }
    _write_manifest(manifest_path, manifest)
    return summary


def main(argv: list[str] | None = None) -> int:
    """Run the explicit offline importer using the decrypted environment."""

    parser = argparse.ArgumentParser(
        description="Import a legacy SQLite response cache into the private artifact store."
    )
    parser.add_argument("source", type=Path, help="absolute legacy SQLite database")
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="absolute owner-only JSON migration manifest",
    )
    args = parser.parse_args(argv)
    load_dotenv()
    store = ArtifactStore.from_env(namespace=DEFAULT_CACHE_NAMESPACE)
    cache = ResponseCache(store, namespace=DEFAULT_CACHE_NAMESPACE)
    summary = import_sqlite_cache(args.source, cache, args.manifest)
    print(
        json.dumps(
            {"counts": summary.counts, "skipped_reasons": summary.skipped_reasons},
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "CacheImportError",
    "ImportSummary",
    "SUPPORTED_OPERATIONS",
    "import_sqlite_cache",
    "main",
]
