"""HTTP client for the participant RAGTIME retrieval service."""

from __future__ import annotations

import math
import os
import sqlite3
import time
import warnings
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import urlsplit

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from requests_ratelimiter import LimiterMixin
from urllib3.util.retry import Retry

from .cache import DEFAULT_CACHE_NAMESPACE, CacheEntry, SQLiteResponseCache
from .models import Document, PipelineResult, SearchHit

DEFAULT_PIPELINE = "ragtime2%20"
DEFAULT_CONTENT_COLLECTION = "ragtime2"
DEFAULT_SEARCH_CACHE_TTL_SECONDS: float | None = None
DEFAULT_CONTENT_CACHE_TTL_SECONDS: float | None = None
DEFAULT_TIMEOUT = (5.0, 60.0)
DEFAULT_REQUESTS_PER_SECOND = 1.0
RETRIABLE_STATUS_CODES = (429, 500, 502, 503, 504)
T = TypeVar("T")


class RagtimeError(RuntimeError):
    """Base exception for this client."""


class RagtimeConfigurationError(RagtimeError):
    """Raised for missing or unsafe client configuration."""


class RagtimeApiError(RagtimeError):
    """Raised when an HTTP request fails."""


class RagtimeProtocolError(RagtimeError):
    """Raised when the service returns an unexpected payload."""


def _canonical_authorization(token: str) -> str:
    value = token.strip()
    if "\r" in value or "\n" in value:
        raise RagtimeConfigurationError("bearer token contains a newline")
    if value.lower().startswith("bearer "):
        value = value[7:].strip()
    if not value or any(character.isspace() for character in value):
        raise RagtimeConfigurationError("bearer token is empty or contains whitespace")
    return f"Bearer {value}"


def _optional_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _ttl_from_env(name: str, default: float | None) -> float | None:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default
    if raw_value.lower() in {"none", "off", "disabled"}:
        return None
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise RagtimeConfigurationError(f"{name} must be a number or 'none'") from exc
    _validate_ttl(name, value)
    return value


def _positive_float_from_env(name: str, default: float) -> float:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise RagtimeConfigurationError(
            f"{name} must be a finite positive number"
        ) from exc
    if not math.isfinite(value) or value <= 0:
        raise RagtimeConfigurationError(f"{name} must be a finite positive number")
    return value


def _validate_ttl(name: str, value: float | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RagtimeConfigurationError(f"{name} must be a number or None")
    if not math.isfinite(float(value)) or value < 0:
        raise RagtimeConfigurationError(f"{name} must be finite and non-negative")


class _RateLimitedSession(LimiterMixin, requests.Session):
    """Requests session with proactive, per-host leaky-bucket limiting."""


def _rate_limited_session(requests_per_second: float) -> requests.Session:
    # Retries are deliberately performed in RagtimeClient._post so every wire
    # attempt passes through LimiterMixin.send() and acquires its own token.
    adapter = HTTPAdapter(max_retries=0)
    session = _RateLimitedSession(per_second=requests_per_second)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


class RagtimeClient:
    """Reusable synchronous RAGTIME client with a shared local cache."""

    def __init__(
        self,
        base_url: str,
        bearer_token: str,
        *,
        cache: SQLiteResponseCache | None = None,
        cache_path: str | os.PathLike[str] | None = None,
        cache_namespace: str = DEFAULT_CACHE_NAMESPACE,
        use_cache: bool = True,
        search_cache_ttl_seconds: float | None = DEFAULT_SEARCH_CACHE_TTL_SECONDS,
        content_cache_ttl_seconds: float | None = DEFAULT_CONTENT_CACHE_TTL_SECONDS,
        timeout: float | tuple[float, float] = DEFAULT_TIMEOUT,
        retries: int = 3,
        requests_per_second: float = DEFAULT_REQUESTS_PER_SECOND,
        session: requests.Session | None = None,
        allow_insecure_http: bool = False,
    ) -> None:
        if not isinstance(use_cache, bool):
            raise RagtimeConfigurationError("use_cache must be a boolean")
        if not isinstance(allow_insecure_http, bool):
            raise RagtimeConfigurationError("allow_insecure_http must be a boolean")
        normalized_base_url = base_url.strip().rstrip("/")
        parsed_url = urlsplit(normalized_base_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise RagtimeConfigurationError("base URL must be an absolute HTTP(S) URL")
        if (
            parsed_url.username is not None
            or parsed_url.password is not None
            or parsed_url.query
            or parsed_url.fragment
        ):
            raise RagtimeConfigurationError(
                "base URL must not contain credentials, a query, or a fragment"
            )
        if parsed_url.scheme != "https" and not allow_insecure_http:
            raise RagtimeConfigurationError("base URL must use HTTPS")
        if retries < 0:
            raise RagtimeConfigurationError("retries must be non-negative")
        if (
            isinstance(requests_per_second, bool)
            or not isinstance(requests_per_second, (int, float))
            or not math.isfinite(float(requests_per_second))
            or requests_per_second <= 0
        ):
            raise RagtimeConfigurationError(
                "requests_per_second must be a finite positive number"
            )
        _validate_ttl("search cache TTL", search_cache_ttl_seconds)
        _validate_ttl("content cache TTL", content_cache_ttl_seconds)

        self.base_url = normalized_base_url
        self._authorization = _canonical_authorization(bearer_token)
        self.timeout = timeout
        self.search_cache_ttl_seconds = search_cache_ttl_seconds
        self.content_cache_ttl_seconds = content_cache_ttl_seconds
        self.requests_per_second = float(requests_per_second)
        self.retries = retries
        # An injected session is an explicit transport override. Preserve its
        # behavior instead of layering this client's limiter or retries on it.
        self._request_retries = retries if session is None else 0
        self._session = (
            session
            if session is not None
            else _rate_limited_session(self.requests_per_second)
        )
        self._owns_session = session is None

        if cache is not None and cache_path is not None:
            raise RagtimeConfigurationError("provide cache or cache_path, not both")
        if not use_cache and cache is not None:
            raise RagtimeConfigurationError(
                "cache cannot be provided when use_cache is false"
            )
        if not use_cache:
            self.cache = None
        elif cache is not None:
            self.cache = cache
        else:
            try:
                self.cache = SQLiteResponseCache(cache_path, namespace=cache_namespace)
            except (OSError, sqlite3.Error, ValueError) as exc:
                raise RagtimeConfigurationError(
                    f"unable to initialize the RAGTIME cache: {exc}"
                ) from None

    @classmethod
    def from_env(cls, **overrides: Any) -> "RagtimeClient":
        """Load the already-decrypted root ``.env`` and build a client."""

        load_dotenv()
        base_url = os.getenv("RAGTIME_BASE_URL", "")
        bearer_token = os.getenv("RAGTIME_BEARER_TOKEN", "")
        if not base_url or not bearer_token:
            raise RagtimeConfigurationError(
                "missing RAGTIME_BASE_URL or RAGTIME_BEARER_TOKEN; decrypt config/secrets.sops.env first"
            )

        options: dict[str, Any] = {
            "cache_namespace": os.getenv(
                "RAGTIME_CACHE_NAMESPACE", DEFAULT_CACHE_NAMESPACE
            ),
            "search_cache_ttl_seconds": _ttl_from_env(
                "RAGTIME_SEARCH_CACHE_TTL_SECONDS",
                DEFAULT_SEARCH_CACHE_TTL_SECONDS,
            ),
            "content_cache_ttl_seconds": _ttl_from_env(
                "RAGTIME_CONTENT_CACHE_TTL_SECONDS",
                DEFAULT_CONTENT_CACHE_TTL_SECONDS,
            ),
            "requests_per_second": _positive_float_from_env(
                "RAGTIME_REQUESTS_PER_SECOND", DEFAULT_REQUESTS_PER_SECOND
            ),
        }
        cache_path = os.getenv("RAGTIME_CACHE_PATH", "").strip()
        if cache_path:
            options["cache_path"] = Path(cache_path).expanduser()
        options.update(overrides)
        return cls(base_url, bearer_token, **options)

    @property
    def cache_path(self) -> Path | None:
        return None if self.cache is None else self.cache.path

    def close(self) -> None:
        if self._owns_session:
            self._session.close()

    def __enter__(self) -> "RagtimeClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _cache_get(
        self,
        *,
        operation: str,
        request: Mapping[str, Any],
        ttl_seconds: float | None,
        refresh: bool,
    ) -> CacheEntry | None:
        if self.cache is None or refresh:
            return None
        try:
            return self.cache.get(
                base_url=self.base_url,
                operation=operation,
                request=request,
                ttl_seconds=ttl_seconds,
            )
        except (OSError, sqlite3.Error) as exc:
            warnings.warn(
                f"RAGTIME cache read failed; requesting upstream ({exc.__class__.__name__})",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

    def _cache_put(
        self,
        *,
        operation: str,
        request: Mapping[str, Any],
        response: Mapping[str, Any],
    ) -> float | None:
        if self.cache is None:
            return None
        try:
            return self.cache.put(
                base_url=self.base_url,
                operation=operation,
                request=request,
                response=response,
            )
        except (OSError, sqlite3.Error) as exc:
            warnings.warn(
                f"RAGTIME cache write failed; returning live response ({exc.__class__.__name__})",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

    def _cache_delete(
        self,
        *,
        operation: str,
        request: Mapping[str, Any],
    ) -> None:
        if self.cache is None:
            return
        try:
            self.cache.delete(
                base_url=self.base_url,
                operation=operation,
                request=request,
            )
        except (OSError, sqlite3.Error) as exc:
            warnings.warn(
                f"RAGTIME cache eviction failed; refreshing upstream ({exc.__class__.__name__})",
                RuntimeWarning,
                stacklevel=2,
            )

    def _post(self, endpoint: str, request: Mapping[str, Any]) -> dict[str, Any]:
        response: requests.Response | Any | None = None
        error: requests.RequestException | None = None
        for attempt in range(self._request_retries + 1):
            try:
                response = self._session.post(
                    f"{self.base_url}{endpoint}",
                    headers={
                        "Authorization": self._authorization,
                        "Content-Type": "application/json",
                    },
                    json=dict(request),
                    timeout=self.timeout,
                )
                response.raise_for_status()
                error = None
                break
            except requests.RequestException as exc:
                error = exc
                status_code = getattr(
                    getattr(exc, "response", None), "status_code", None
                )
                retriable = status_code in RETRIABLE_STATUS_CODES or status_code is None
                if attempt >= self._request_retries or not retriable:
                    break

                retry = Retry(
                    backoff_factor=0.5,
                    respect_retry_after_header=True,
                )
                retry_after = None
                if response is not None and hasattr(response, "headers"):
                    retry_after = retry.get_retry_after(response)
                delay = (
                    retry_after
                    if retry_after is not None
                    else 0.5 * (2**attempt)
                )
                time.sleep(delay)

        if error is not None:
            status_code = getattr(getattr(error, "response", None), "status_code", None)
            suffix = (
                f" HTTP {status_code}"
                if status_code is not None
                else f" {error.__class__.__name__}"
            )
            raise RagtimeApiError(f"RAGTIME {endpoint} request failed:{suffix}") from None

        assert response is not None

        try:
            payload = response.json()
        except ValueError:
            raise RagtimeProtocolError(
                f"RAGTIME {endpoint} returned invalid JSON"
            ) from None
        if not isinstance(payload, dict):
            raise RagtimeProtocolError(
                f"RAGTIME {endpoint} returned a non-object JSON payload"
            )
        return payload

    def _validated_request(
        self,
        *,
        operation: str,
        endpoint: str,
        request: Mapping[str, Any],
        ttl_seconds: float | None,
        refresh: bool,
        validator: Callable[[Mapping[str, Any]], T],
    ) -> tuple[T, bool, float | None]:
        entry = self._cache_get(
            operation=operation,
            request=request,
            ttl_seconds=ttl_seconds,
            refresh=refresh,
        )
        if entry is not None:
            try:
                return validator(entry.payload), True, entry.stored_at
            except RagtimeProtocolError:
                self._cache_delete(operation=operation, request=request)

        payload = self._post(endpoint, request)
        validated = validator(payload)
        stored_at = self._cache_put(
            operation=operation,
            request=request,
            response=payload,
        )
        return validated, False, stored_at

    @staticmethod
    def _parse_pipeline_payload(
        payload: Mapping[str, Any],
        *,
        query: str,
        pipeline: str,
    ) -> tuple[
        tuple[SearchHit, ...],
        bool | None,
        bool | None,
        float | None,
        dict[str, Any],
    ]:
        echoed_query = payload.get("query")
        if echoed_query is not None and echoed_query != query:
            raise RagtimeProtocolError("RAGTIME /pipeline echoed a different query")
        echoed_pipeline = payload.get("pipeline")
        if echoed_pipeline is not None and echoed_pipeline != pipeline:
            raise RagtimeProtocolError("RAGTIME /pipeline echoed a different pipeline")

        scores = payload.get("scores")
        if not isinstance(scores, dict):
            raise RagtimeProtocolError(
                "RAGTIME /pipeline response is missing a scores object"
            )
        ranked_scores: list[tuple[str, float]] = []
        for document_id, score in scores.items():
            if not isinstance(document_id, str) or not document_id:
                raise RagtimeProtocolError(
                    "RAGTIME /pipeline returned an invalid document ID"
                )
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise RagtimeProtocolError(
                    "RAGTIME /pipeline returned a non-numeric score"
                )
            numeric_score = float(score)
            if not math.isfinite(numeric_score):
                raise RagtimeProtocolError(
                    "RAGTIME /pipeline returned a non-finite score"
                )
            ranked_scores.append((document_id, numeric_score))
        ranked_scores.sort(key=lambda item: (-item[1], item[0]))
        hits = tuple(
            SearchHit(rank=rank, document_id=document_id, score=score)
            for rank, (document_id, score) in enumerate(ranked_scores, start=1)
        )
        known_keys = {"cached", "pipeline", "processed", "query", "scores", "timestamp"}
        return (
            hits,
            _optional_bool(payload.get("processed")),
            _optional_bool(payload.get("cached")),
            _optional_float(payload.get("timestamp")),
            {key: value for key, value in payload.items() if key not in known_keys},
        )

    @staticmethod
    def _parse_content_payload(
        payload: Mapping[str, Any],
        *,
        document_id: str,
        collection: str,
    ) -> tuple[str, bool | None, float | None, dict[str, Any]]:
        response_id = payload.get("id")
        response_collection = payload.get("collection")
        text = payload.get("text")
        if response_id != document_id:
            raise RagtimeProtocolError(
                "RAGTIME /content returned a different document ID"
            )
        if response_collection != collection:
            raise RagtimeProtocolError(
                "RAGTIME /content returned a different collection"
            )
        if not isinstance(text, str):
            raise RagtimeProtocolError(
                "RAGTIME /content response is missing document text"
            )
        known_keys = {"cached", "collection", "id", "text", "timestamp"}
        return (
            text,
            _optional_bool(payload.get("cached")),
            _optional_float(payload.get("timestamp")),
            {key: value for key, value in payload.items() if key not in known_keys},
        )

    def search(
        self,
        query: str,
        *,
        pipeline: str = DEFAULT_PIPELINE,
        refresh: bool = False,
    ) -> PipelineResult:
        """Run one pipeline query and return deterministically ranked hits."""

        if not isinstance(refresh, bool):
            raise ValueError("refresh must be a boolean")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(pipeline, str) or not pipeline.strip():
            raise ValueError("pipeline must be a non-empty string")
        request = {"pipeline": pipeline, "query": query}
        parsed, local_cache_hit, cache_stored_at = self._validated_request(
            operation="pipeline",
            endpoint="/pipeline",
            request=request,
            ttl_seconds=self.search_cache_ttl_seconds,
            refresh=refresh,
            validator=lambda payload: self._parse_pipeline_payload(
                payload,
                query=query,
                pipeline=pipeline,
            ),
        )
        hits, processed, remote_cached, server_timestamp, extra = parsed
        return PipelineResult(
            query=query,
            pipeline=pipeline,
            hits=hits,
            processed=processed,
            local_cache_hit=local_cache_hit,
            remote_cached=remote_cached,
            server_timestamp=server_timestamp,
            cache_stored_at=cache_stored_at,
            extra=extra,
        )

    def pipeline(
        self,
        query: str,
        *,
        pipeline: str = DEFAULT_PIPELINE,
        refresh: bool = False,
    ) -> PipelineResult:
        """Alias for :meth:`search` matching the HTTP endpoint name."""

        return self.search(query, pipeline=pipeline, refresh=refresh)

    def get_content(
        self,
        document_id: str,
        *,
        collection: str = DEFAULT_CONTENT_COLLECTION,
        refresh: bool = False,
    ) -> Document:
        """Retrieve one full document by its opaque search-result ID."""

        if not isinstance(refresh, bool):
            raise ValueError("refresh must be a boolean")
        if not isinstance(document_id, str) or not document_id:
            raise ValueError("document_id must be a non-empty string")
        if not isinstance(collection, str) or not collection:
            raise ValueError("collection must be a non-empty string")
        request = {"collection": collection, "id": document_id}
        parsed, local_cache_hit, cache_stored_at = self._validated_request(
            operation="content",
            endpoint="/content",
            request=request,
            ttl_seconds=self.content_cache_ttl_seconds,
            refresh=refresh,
            validator=lambda payload: self._parse_content_payload(
                payload,
                document_id=document_id,
                collection=collection,
            ),
        )
        text, remote_cached, server_timestamp, extra = parsed
        return Document(
            collection=collection,
            document_id=document_id,
            text=text,
            local_cache_hit=local_cache_hit,
            remote_cached=remote_cached,
            server_timestamp=server_timestamp,
            cache_stored_at=cache_stored_at,
            extra=extra,
        )
