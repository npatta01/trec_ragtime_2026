"""Process-safe SQLite response cache shared across repository worktrees."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .paths import default_project_cache_dir, ensure_private_cache_dir

CACHE_SCHEMA_VERSION = 1
DEFAULT_CACHE_NAMESPACE = "ragtime2-2026-v1"
DEFAULT_CACHE_FILENAME = "api-v1.sqlite3"


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """A decoded cached response and the time it was stored."""

    payload: dict[str, Any]
    stored_at: float


def _absolute_path(value: str | os.PathLike[str], setting: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{setting} must be an absolute path")
    return path


def default_cache_path() -> Path:
    """Return the project-level API cache shared by every worktree."""

    explicit_path = os.getenv("RAGTIME_CACHE_PATH", "").strip()
    if explicit_path:
        return _absolute_path(explicit_path, "RAGTIME_CACHE_PATH")

    explicit_dir = os.getenv("RAGTIME_CACHE_DIR", "").strip()
    if explicit_dir:
        return (
            _absolute_path(explicit_dir, "RAGTIME_CACHE_DIR") / DEFAULT_CACHE_FILENAME
        )

    return default_project_cache_dir() / DEFAULT_CACHE_FILENAME


class SQLiteResponseCache:
    """Small cache-aside store safe for concurrent local processes."""

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
        return json.dumps(
            value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )

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
        """Return a fresh cache entry, or ``None`` for a miss/expiry/corruption."""

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
                connection.execute(
                    "DELETE FROM responses WHERE cache_key = ?", (cache_key,)
                )
                connection.commit()
                return None
            if not isinstance(payload, dict):
                connection.execute(
                    "DELETE FROM responses WHERE cache_key = ?", (cache_key,)
                )
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
        """Atomically insert or replace one validated response."""

        cache_key, request_json = self._identity(
            base_url=base_url,
            operation=operation,
            request=request,
        )
        response_json = self._canonical_json(response)
        stored_at = self._clock()
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO responses (
                        cache_key,
                        schema_version,
                        namespace,
                        base_url,
                        operation,
                        request_json,
                        response_json,
                        stored_at
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
        """Remove all responses in this cache namespace."""

        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    "DELETE FROM responses WHERE namespace = ?", (self.namespace,)
                )

    def delete(
        self,
        *,
        base_url: str,
        operation: str,
        request: Mapping[str, Any],
    ) -> None:
        """Delete one response after detecting a stale or incompatible schema."""

        cache_key, _ = self._identity(
            base_url=base_url,
            operation=operation,
            request=request,
        )
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    "DELETE FROM responses WHERE cache_key = ?", (cache_key,)
                )
